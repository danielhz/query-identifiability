PYTHON  := python3
PYTEST  := $(PYTHON) -m pytest
RESULTS := results

.PHONY: test test-verbose lint fmt typecheck coverage ci \
        mini e1 e2 e3 ablation e4 analyze \
        crosskg-data crosskg-data-api crosskg-probe crosskg-witness \
        collect-horeka collect-aisa submit-horeka submit-ablation clean

# ---------------------------------------------------------------------------
# Development (local / ARM)
# ---------------------------------------------------------------------------

test:
	$(PYTEST) tests/ -q

test-verbose:
	$(PYTEST) tests/ -v

# Ruff lint check (no auto-fix)
lint:
	ruff check .

# Ruff auto-format + auto-fix imports
fmt:
	ruff format .
	ruff check . --fix

# Mypy static type checking (source packages only; tests are excluded in pyproject.toml)
typecheck:
	$(PYTHON) -m mypy data/ models/ experiments/ analysis/

# pytest-cov coverage report
coverage:
	$(PYTEST) tests/ -q --cov --cov-report=term-missing

# Full CI gate: format check, lint, types, tests
ci: fmt lint typecheck test

# Run all three experiments in smoke-test mode (< 60 s on CPU)
mini: mini-e1 mini-e2 mini-e3 mini-ablation mini-e4

mini-e1:
	$(PYTHON) -m experiments.e1_error_floor --mini --output-dir $(RESULTS)

mini-e2:
	$(PYTHON) -m experiments.e2_capability_jump --mini --output-dir $(RESULTS)

mini-e3:
	$(PYTHON) -m experiments.e3_minaug --mini --output-dir $(RESULTS)

mini-ablation:
	$(PYTHON) -m experiments.e1_ablation --mini --output-dir $(RESULTS)

# ---------------------------------------------------------------------------
# Full experiments (run on cluster, or locally with --device cpu for small scale)
# ---------------------------------------------------------------------------

e1:
	$(PYTHON) -m experiments.e1_error_floor \
	    --n-train 5000 --n-val 500 --n-test 1000 \
	    --n-tuples 20 --n-cert 10 --n-noncert 10 \
	    --epochs 300 --patience 30 --hidden-dim 128 \
	    --seeds 0 1 2 --output-dir $(RESULTS)

e2:
	$(PYTHON) -m experiments.e2_capability_jump \
	    --n-train 5000 --n-val 500 --n-test 1000 \
	    --n-tuples 20 --epochs 300 --patience 30 \
	    --hidden-dim 128 --seeds 0 1 2 --output-dir $(RESULTS)

e3:
	$(PYTHON) -m experiments.e3_minaug \
	    --n-trials 2000 \
	    --n-attrs-range 4 6 8 10 12 15 \
	    --n-fds-range 0 2 4 6 10 \
	    --output-dir $(RESULTS)

ablation:
	$(PYTHON) -m experiments.e1_ablation \
	    --n-train 5000 --n-val 500 --n-test 1000 --n-tuples 20 \
	    --epochs 300 --patience 30 --hidden-dim 128 \
	    --seeds 0 1 2 --device cuda --output-dir $(RESULTS)

e4:
	$(PYTHON) -m experiments.e4_realworld \
	    --n-train 5000 --n-val 500 --n-test 1000 \
	    --feature-domain 16 \
	    --epochs 300 --patience 30 --hidden-dim 128 \
	    --seeds 0 1 2 --device cuda --output-dir $(RESULTS)

mini-e4:
	$(PYTHON) -m experiments.e4_realworld --mini --output-dir $(RESULTS)

# ---------------------------------------------------------------------------
# Analysis (run locally after collecting results)
# ---------------------------------------------------------------------------

analyze: analyze-e1 analyze-e2 analyze-e3 analyze-ablation analyze-e4

analyze-e1:
	$(PYTHON) -m analysis.plot_error_floor     --results-dir $(RESULTS) --latex

analyze-e2:
	$(PYTHON) -m analysis.plot_capability_jump --results-dir $(RESULTS)

analyze-e3:
	$(PYTHON) -m analysis.plot_minaug          --results-dir $(RESULTS) --latex

analyze-ablation:
	$(PYTHON) -m analysis.plot_ablation        --results-dir $(RESULTS) --latex

analyze-e4:
	$(PYTHON) -m analysis.plot_realworld       --results-dir $(RESULTS) --latex

# ---------------------------------------------------------------------------
# Cross-source probe: OpenAlex x DBLP (third real dataset — real witnesses)
#   Two independent CS bibliographies for the same papers, joined on DOI.
#   Measures the natural certified/uncertified split on real values:
#   author-count genuinely conflicts (~1.3%, real witnesses); title mostly
#   agrees. Provenance: dblp.org/xml/dblp.xml.gz + api.openalex.org.
# ---------------------------------------------------------------------------

# Build the joined corpus from the DBLP bulk dump (robust; ~1GB download + parse).
# Writes data/raw/crosskg_dblp/{dblp,openalex}.csv (git-ignored raw data).
crosskg-data:
	mkdir -p data/raw/crosskg_dblp
	curl -fSL -o data/raw/crosskg_dblp/dblp.xml.gz https://dblp.org/xml/dblp.xml.gz
	$(PYTHON) -m data.build_crosskg_dblp --sample 6000 --seed 0

# Alternative: build a small corpus via the DBLP search API (no dump; rate-limited).
crosskg-data-api:
	$(PYTHON) -m data.download_crosskg_dblp --max-dois 2500

# Run the fail-fast probe (certified/uncertified split) on the joined corpus.
crosskg-probe:
	$(PYTHON) -m experiments.e_crosskg_probe

# Real-witness discovery rate (RQ2 CDF) on the joined corpus.
crosskg-witness:
	$(PYTHON) -m experiments.e_witness_discovery --dataset crosskg --output-dir $(RESULTS)

# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

submit-horeka: submit-ablation
	bash cluster/submit_all.sh

submit-ablation:
	bash cluster/submit_ablation.sh

collect-horeka:
	bash cluster/collect_results.sh horeka

collect-aisa:
	bash cluster/collect_results.sh aisa

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
