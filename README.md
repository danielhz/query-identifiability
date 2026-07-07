# query-identifiability

Tools and experiments for **certifying when a Boolean conjunctive query is identifiable from interface-visible views**, and for prescribing the minimum schema augmentation when it is not.

Given a set of source views with designated overlaps Ω and functional-dependency (FD) interface laws Σ, a query is *identifiable* when every legal world consistent with the interface evidence returns the same answer. This repository implements:

- **CheckCert** — a sufficient identifiability certificate based on the Σ-closure of the designated overlaps (`data/utils.py`).
- **Greedy-MinAug** — a greedy `H(|S|)`-approximation that prescribes the minimum overlap augmentation needed to certify a non-certified query (`data/utils.py`, `experiments/e_minaug_*`).
- **Witness finding** — exact and sampling-based construction of explicit non-identifiability witnesses (`data/witness.py`, `data/exhaustive.py`).

The companion Lean 4 formalization of the theoretical results lives in a separate repository: <https://github.com/danielhz/MultiViewIdentifiability>.

## Install

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # runtime + tests
pip install -r requirements-dev.txt    # ruff + mypy (for `make ci`)
```

## Quick start

```sh
make mini      # all experiments in smoke-test mode (<60 s, CPU)
make ci        # format check, lint, type-check, tests
make analyze   # regenerate figures/tables from results/
```

## Experiments

Experiments are Python modules under `experiments/` (run as `python -m experiments.<name>`); analysis under `analysis/`; dataset loaders under `data/`; results as timestamped JSON in `results/`.

| Module | What it measures |
|--------|------------------|
| `e_cert_benchmark` / `e_exhaustive_check` | Exactness of the closure certificate vs. an exhaustive oracle |
| `e_realworld_witnesses` | Certified/uncertified split and witnesses on real datasets |
| `e_witness_discovery` | Empirical witness-discovery rate vs. corpus size |
| `e_minaug_multiatom` / `e_minaug_realworld` | Greedy-MinAug approximation quality and real-schema prescriptions |
| `e_scalability` | CheckCert / Greedy-MinAug runtime vs. schema size |

### Datasets

Real datasets download into `data/raw/`; tests and CI use in-memory mock generators (`make_mock_dataset()`), so **no network is needed for tests**.

- **BibInteg** — scholarly records via the OpenAlex API (`data/download_openalex.py`).
- **CrossKG-DBLP** — OpenAlex × DBLP, two independent bibliographies joined on DOI (`data/build_crosskg_dblp.py`; `make crosskg-data`).
- **WDC-Product** — product schema illustration (`data/download_wdc.py`).
- **WikiScholar** — scholarly articles extracted from Wikidata (`data/wikidata_extraction_task.md`).

## Repository layout

```
data/         dataset loaders, FD/closure core, witness finders
experiments/  experiment drivers (python -m experiments.<name>)
analysis/     figure/table generation (pgfdata CSVs + matplotlib PDFs)
cluster/      optional SLURM submission scripts
tests/        pytest suite (mock data only)
results/      timestamped JSON experiment outputs
```

## Authors

- Ratan Bahadur Thapa
- Daniel Hernández

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this software, please cite: Thapa & Hernández, *Identifiability of Relational Queries in Multi-View Pretraining*, arXiv:2607.04735 (2026). Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

Archived at DaRUS: <https://doi.org/10.18419/DARUS-6292>.
