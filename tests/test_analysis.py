"""
Smoke tests for the analysis scripts.

Each test:
1. Runs the corresponding experiment in --mini mode to produce a JSON file.
2. Calls the plot function directly against that output.
3. Checks that the figure file was written and has non-zero size.
4. Checks the LaTeX helper (where applicable) for basic structural correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.utils import grouped_stats, latest_result, load_results, to_latex_table
from experiments.e1_error_floor import main as e1_main
from experiments.e2_capability_jump import main as e2_main
from experiments.e3_minaug import main as e3_main

# ---------------------------------------------------------------------------
# Fixtures: run experiments once per session and cache in a shared tmp dir
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("results")
    # Run all three mini experiments
    e1_main(["--mini", "--output-dir", str(d)])
    e2_main(["--mini", "--output-dir", str(d)])
    e3_main(["--mini", "--output-dir", str(d)])
    return d


# ---------------------------------------------------------------------------
# E1 — Error floor plot
# ---------------------------------------------------------------------------


class TestPlotErrorFloor:
    def test_figure_created(self, results_dir, tmp_path):
        from analysis.plot_error_floor import plot

        src = latest_result(results_dir, "e1_error_floor")
        records, _ = load_results(src)
        out = tmp_path / "error_floor.pdf"
        fig = plot(records, out)
        assert fig is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_figure_png_variant(self, results_dir, tmp_path):
        from analysis.plot_error_floor import plot

        src = latest_result(results_dir, "e1_error_floor")
        records, _ = load_results(src)
        out = tmp_path / "error_floor.png"
        plot(records, out)
        assert out.exists()

    def test_latex_table_structure(self, results_dir):
        from analysis.plot_error_floor import latex_table

        src = latest_result(results_dir, "e1_error_floor")
        records, _ = load_results(src)
        tex = latex_table(records)
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex
        assert "tab:certificate" in tex

    def test_latex_table_has_all_architectures(self, results_dir):
        from analysis.plot_error_floor import latex_table
        from analysis.utils import ARCH_LABEL

        src = latest_result(results_dir, "e1_error_floor")
        records, _ = load_results(src)
        tex = latex_table(records)
        models_used = {r["model"] for r in records}
        for m in models_used:
            assert ARCH_LABEL[m] in tex

    def test_cli_runs(self, results_dir, tmp_path):
        from analysis.plot_error_floor import main

        out = tmp_path / "ef.pdf"
        main(["--input", str(latest_result(results_dir, "e1_error_floor")), "--output", str(out)])
        assert out.exists()

    def test_cli_latex_flag(self, results_dir, tmp_path, capsys):
        from analysis.plot_error_floor import main

        out = tmp_path / "ef2.pdf"
        main(
            [
                "--input",
                str(latest_result(results_dir, "e1_error_floor")),
                "--output",
                str(out),
                "--latex",
            ]
        )
        captured = capsys.readouterr().out
        assert "tab:certificate" in captured


# ---------------------------------------------------------------------------
# E2 — Capability jump plot
# ---------------------------------------------------------------------------


class TestPlotCapabilityJump:
    def test_figure_created(self, results_dir, tmp_path):
        from analysis.plot_capability_jump import plot

        src = latest_result(results_dir, "e2_capability_jump")
        records, _ = load_results(src)
        out = tmp_path / "capability_jump.pdf"
        fig = plot(records, out)
        assert fig is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_figure_png_variant(self, results_dir, tmp_path):
        from analysis.plot_capability_jump import plot

        src = latest_result(results_dir, "e2_capability_jump")
        records, _ = load_results(src)
        out = tmp_path / "capability_jump.png"
        plot(records, out)
        assert out.exists()

    def test_cli_runs(self, results_dir, tmp_path):
        from analysis.plot_capability_jump import main

        out = tmp_path / "cj.pdf"
        main(
            [
                "--input",
                str(latest_result(results_dir, "e2_capability_jump")),
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_cert_step_detection(self, results_dir):
        from analysis.plot_capability_jump import _cert_step

        src = latest_result(results_dir, "e2_capability_jump")
        records, _ = load_results(src)
        step = _cert_step(records)
        assert step is not None
        # In mini mode we only run step 0 and the last (certified) step
        last_step = max(r["step"] for r in records)
        assert step == last_step

    def test_steps_monotone_in_overlap_size(self, results_dir):
        src = latest_result(results_dir, "e2_capability_jump")
        records, _ = load_results(src)
        # Each step has exactly one aug_overlap_size; check they're non-decreasing
        step_size = {}
        for r in records:
            step_size[r["step"]] = r["aug_overlap_size"]
        sizes = [step_size[s] for s in sorted(step_size)]
        for a, b in zip(sizes, sizes[1:]):
            assert a <= b


# ---------------------------------------------------------------------------
# E3 — MinAug plot
# ---------------------------------------------------------------------------


class TestPlotMinAug:
    def test_figure_created(self, results_dir, tmp_path):
        from analysis.plot_minaug import plot

        src = latest_result(results_dir, "e3_minaug")
        records, _ = load_results(src)
        out = tmp_path / "minaug.pdf"
        fig = plot(records, out)
        assert fig is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_figure_has_two_panels(self, results_dir, tmp_path):
        from analysis.plot_minaug import plot

        src = latest_result(results_dir, "e3_minaug")
        records, _ = load_results(src)
        out = tmp_path / "minaug_panels.pdf"
        fig = plot(records, out)
        assert len(fig.axes) == 2

    def test_latex_table_structure(self, results_dir):
        from analysis.plot_minaug import latex_stats

        src = latest_result(results_dir, "e3_minaug")
        records, _ = load_results(src)
        tex = latex_stats(records)
        assert r"\begin{table}" in tex
        assert "tab:minaug" in tex

    def test_cli_runs(self, results_dir, tmp_path):
        from analysis.plot_minaug import main

        out = tmp_path / "ma.pdf"
        main(["--input", str(latest_result(results_dir, "e3_minaug")), "--output", str(out)])
        assert out.exists()

    def test_cli_latex_flag(self, results_dir, tmp_path, capsys):
        from analysis.plot_minaug import main

        out = tmp_path / "ma2.pdf"
        main(
            [
                "--input",
                str(latest_result(results_dir, "e3_minaug")),
                "--output",
                str(out),
                "--latex",
            ]
        )
        assert "tab:minaug" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# analysis/utils helpers
# ---------------------------------------------------------------------------


class TestAnalysisUtils:
    def test_grouped_stats_mean_in_range(self, results_dir):
        src = latest_result(results_dir, "e1_error_floor")
        records, _ = load_results(src)
        stats = grouped_stats(records, ["model", "query_type"])
        for (model, qtype), (mean, std) in stats.items():
            assert 0.0 <= mean <= 1.0
            assert std >= 0.0

    def test_to_latex_table_structure(self):
        tex = to_latex_table(
            rows=[[0.9, 0.5], [0.8, 0.48]],
            col_headers=["Certified", "Non-cert."],
            row_headers=["MLP", "GNN-OG"],
            caption="Test caption",
            label="tab:test",
        )
        assert "tabular" in tex
        assert "MLP" in tex
        assert "GNN-OG" in tex
        assert "tab:test" in tex

    def test_latest_result_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            latest_result(tmp_path, "e1_error_floor")
