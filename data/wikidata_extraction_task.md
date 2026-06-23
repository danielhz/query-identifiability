# Wikidata Extraction Task

## Context

This file is instructions for an agent running on the server where the Wikidata dump
has already been downloaded. The goal is to extract a subset of scholarly articles
from Wikidata and produce three CSV files that serve as a real-world benchmark
dataset ("WikiScholar") for this project.

The project studies *query identifiability* in multi-source data integration: a query
is identifiable when every legal world consistent with the interface evidence returns
the same answer. The key mechanism is the *Σ-closure* of designated overlaps under
functional-dependency (FD) interface laws.

## What we need

We need three CSV files representing three "views" of the same set of scholarly
articles, where each view exposes a different set of attributes beyond a shared
overlap:

| File | View name | Shared (overlap) columns | View-private column |
|------|-----------|--------------------------|---------------------|
| `crossref.csv` | Crossref / publisher | title, author, year | venue |
| `arxiv.csv` | arXiv | title, author, year | subject\_category |
| `citations.csv` | Citation index | title, author, year | n\_citations |

**All three files must contain the same set of papers**, projected onto different
attribute subsets. The shared key `(title, author, year)` is the overlap, and the
FD interface laws are:

```
{title, author, year} → venue
{title, author, year} → subject_category
{title, author, year} → n_citations
```

These FDs hold by construction (a paper has one primary venue, one arXiv category,
one citation count at extraction time), so every query whose footprint is covered by
the Σ-closure of `{title, author, year}` is certified — which makes all three queries
in the benchmark certified.

## Target size

- At least **10,000 papers** (to support n\_train=5000, n\_val=500, n\_test=1000 with
  a comfortable margin)
- Papers must have **all four attributes** (venue, subject\_category, n\_citations)
  non-null and non-empty — records missing any of these should be dropped

## Wikidata properties to use

Extract items that are instances of **scholarly article** (`P31 = Q13442814`) and
have all of the following properties:

| Column | Wikidata property | Notes |
|--------|-------------------|-------|
| title | `P1476` (title) or `rdfs:label` | Use English label; strip quotes |
| author | `P50` (author) | Take first author's name; if multiple, join with "; " |
| year | `P577` (publication date) | Extract 4-digit year |
| venue | `P1433` (published in) | English label of the journal/conference |
| subject\_category | `P921` (main subject) OR `P356`-linked arXiv category | English label of first subject |
| n\_citations | `P2860` (cites) count OR `P1082` (number of participants) as fallback | Count of outgoing P2860 statements |

If `P921` is absent, use the first word of `rdfs:label` of the item linked via
`P2860` as a coarse subject proxy. If `n_citations` cannot be determined, use the
count of `P2860` statements on the item (number of references/citations it makes).

## Output format

Three CSV files with the following columns and naming:

```
data/raw/wikischolar/crossref.csv   → title, author, year, venue
data/raw/wikischolar/arxiv.csv      → title, author, year, subject_category
data/raw/wikischolar/citations.csv  → title, author, year, n_citations
```

All three CSVs must have exactly the same number of rows and the same ordering
(row i in crossref.csv, arxiv.csv, and citations.csv corresponds to the same paper).

## Suggested extraction approach

If the Wikidata dump is in JSON format (`latest-all.json.gz`):

```python
import gzip, json

with gzip.open("latest-all.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip().rstrip(",")
        if not (line.startswith("{") and line.endswith("}")):
            continue
        item = json.loads(line)
        # filter: instance of scholarly article
        claims = item.get("claims", {})
        p31 = [s["mainsnak"]["datavalue"]["value"]["id"]
               for s in claims.get("P31", [])
               if s["mainsnak"].get("snaktype") == "value"]
        if "Q13442814" not in p31:
            continue
        # extract P1476 (title), P50 (author), P577 (year),
        #         P1433 (venue), P921 (subject), P2860 count
        ...
```

If the dump is in RDF/Turtle or N-Triples format, use a SPARQL query against a
local Blazegraph/Fuseki instance, or use `rdflib` to stream the triples.

Alternatively, if a local SPARQL endpoint is available, this query retrieves the
needed data:

```sparql
SELECT ?title ?authorLabel ?year ?venueLabel ?subjectLabel (COUNT(?cited) AS ?n_citations)
WHERE {
  ?paper wdt:P31 wd:Q13442814 ;
         wdt:P1476 ?title ;
         wdt:P50   ?author ;
         wdt:P577  ?pubdate ;
         wdt:P1433 ?venue .
  OPTIONAL { ?paper wdt:P921 ?subject }
  OPTIONAL { ?paper wdt:P2860 ?cited }
  BIND(YEAR(?pubdate) AS ?year)
  FILTER(?year >= 2010 && ?year <= 2024)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
GROUP BY ?title ?authorLabel ?year ?venueLabel ?subjectLabel
HAVING (COUNT(?cited) >= 0)
LIMIT 50000
```

## Saving the output

Save the three CSV files to `data/raw/wikischolar/` within this repository
(the directory is gitignored for raw data; processed results go elsewhere).

After extraction, verify the output with:

```bash
python -m data.wikischolar --info --data-dir data/raw/wikischolar
```

(You will need to create `data/wikischolar.py` analogously to `data/bibinteg.py`,
following the same schema: 7 attributes, 3 views, overlap `{0,1,2}`, FDs as above.)

## Contact / questions

If the Wikidata dump format is different from what is described here, or if any
of the properties are missing for the bulk of records, please adjust the extraction
strategy and document the changes in this file before proceeding.
