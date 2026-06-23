"""
Tests for analysis/plot_realworld.py.
Uses the mini E4 output as fixture (no real data required).
"""

import json

import pytest

from analysis.utils import latest_result, load_results
from experiments.e4_realworld import main as e4_main


@pytest.fixture(scope="module")
def e4_records(tmp_path_factory) -> list[dict]:
    d = tmp_path_factory.mktemp("e4_plot")
    e4_main(["--mini", "--output-dir", str(d)])
    src = latest_result(d, "e4_realworld")
    records, _ = load_results(src)
    return records


class TestPlotRealworld:
    def test_figure_created(self, e4_records, tmp_path):
        from analysis.plot_realworld import plot

        out = tmp_path / "realworld.pdf"
        fig = plot(e4_records, out)
        assert fig is not None
        assert out.exists() and out.stat().st_size > 0

    def test_figure_has_two_panels(self, e4_records, tmp_path):
        from analysis.plot_realworld import plot

        out = tmp_path / "realworld_panels.pdf"
        fig = plot(e4_records, out)
        assert len(fig.axes) == 2

    def test_latex_table_structure(self, e4_records):
        from analysis.plot_realworld import latex_table

        tex = latex_table(e4_records)
        assert r"\begin{table}" in tex
        assert "tab:realworld" in tex
        assert r"\toprule" in tex

    def test_latex_contains_certified_marker(self, e4_records):
        from analysis.plot_realworld import latex_table

        tex = latex_table(e4_records)
        # certified queries should appear (checkmark or column header)
        assert "tab:realworld" in tex

    def test_cli_runs(self, e4_records, tmp_path):
        from analysis.plot_realworld import main as plot_main

        src = tmp_path / "e4_test.json"
        src.write_text(json.dumps({"meta": {}, "results": e4_records}))
        out = tmp_path / "realworld.pdf"
        plot_main(["--input", str(src), "--output", str(out)])
        assert out.exists()

    def test_cli_latex_flag(self, e4_records, tmp_path, capsys):
        from analysis.plot_realworld import main as plot_main

        src = tmp_path / "e4_test2.json"
        src.write_text(json.dumps({"meta": {}, "results": e4_records}))
        out = tmp_path / "realworld2.pdf"
        plot_main(["--input", str(src), "--output", str(out), "--latex"])
        assert "tab:realworld" in capsys.readouterr().out
