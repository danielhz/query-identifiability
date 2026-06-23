"""
Tests for the ablation experiment and its analysis script.
"""

import json

import pytest

from analysis.utils import latest_result, load_results
from experiments.e1_ablation import (
    _ALL_FDS,
    QUERY,
    _make_config,
    _rho_levels,
)
from experiments.e1_ablation import (
    main as ablation_main,
)

# ---------------------------------------------------------------------------
# Unit tests: schema and ρ-level logic
# ---------------------------------------------------------------------------


class TestRhoLevels:
    def test_length(self):
        levels = _rho_levels(4)
        assert len(levels) == 5  # k = 0,1,2,3,4

    def test_rho_values(self):
        levels = _rho_levels(4)
        rhos = [r for r, _ in levels]
        assert rhos == pytest.approx([0.0, 0.25, 0.50, 0.75, 1.0])

    def test_fds_grow_monotonically(self):
        levels = _rho_levels(4)
        for i in range(len(levels) - 1):
            _, fds_now = levels[i]
            _, fds_next = levels[i + 1]
            assert len(fds_now) < len(fds_next)

    def test_prefix_property(self):
        levels = _rho_levels(4)
        for k, (_, fds) in enumerate(levels):
            assert fds == list(_ALL_FDS[:k])


class TestCertification:
    def test_not_certified_at_rho_0(self):
        config = _make_config([])
        assert not QUERY.is_certified(config)

    def test_not_certified_at_intermediate_rho(self):
        for k in range(1, 4):
            config = _make_config(_ALL_FDS[:k])
            assert not QUERY.is_certified(config), f"Should not be certified at k={k}"

    def test_certified_at_rho_1(self):
        config = _make_config(_ALL_FDS)
        assert QUERY.is_certified(config)

    def test_aug_overlap_grows(self):
        from data.utils import augmented_overlap
        from experiments.e1_ablation import _OVERLAP_SCHEMAS

        prev_size = 0
        for k in range(len(_ALL_FDS) + 1):
            aug = augmented_overlap(_OVERLAP_SCHEMAS[0], _ALL_FDS[:k])
            assert len(aug) >= prev_size
            prev_size = len(aug)


# ---------------------------------------------------------------------------
# End-to-end: experiment produces valid JSON
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ablation_results(tmp_path_factory) -> tuple[list[dict], dict]:
    d = tmp_path_factory.mktemp("ablation")
    ablation_main(["--mini", "--output-dir", str(d)])
    src = latest_result(d, "e1_ablation")
    return load_results(src)


class TestAblationOutput:
    def test_json_written(self, tmp_path):
        ablation_main(["--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e1_ablation_*.json"))
        assert len(files) == 1

    def test_meta_fields(self, ablation_results):
        _, meta = ablation_results
        assert meta["experiment"] == "e1_ablation"
        assert meta["n_fds_total"] == 4
        assert set(meta["rho_levels"]) == {0.0, 1.0}  # mini: only endpoints

    def test_result_keys(self, ablation_results):
        records, _ = ablation_results
        required = {
            "model",
            "rho",
            "n_fds_revealed",
            "n_fds_total",
            "certified",
            "aug_overlap",
            "aug_overlap_size",
            "feature_dim",
            "seed",
            "test_accuracy",
        }
        for r in records:
            assert required <= r.keys()

    def test_rho_0_not_certified(self, ablation_results):
        records, _ = ablation_results
        for r in [r for r in records if r["rho"] == 0.0]:
            assert r["certified"] is False

    def test_rho_1_certified(self, ablation_results):
        records, _ = ablation_results
        for r in [r for r in records if r["rho"] == 1.0]:
            assert r["certified"] is True

    def test_feature_dim_grows_with_rho(self, ablation_results):
        records, _ = ablation_results
        # For any single architecture: feature_dim at rho=1 > feature_dim at rho=0
        rho0 = [r for r in records if r["rho"] == 0.0 and r["model"] == "mlp"]
        rho1 = [r for r in records if r["rho"] == 1.0 and r["model"] == "mlp"]
        if rho0 and rho1:
            assert rho0[0]["feature_dim"] < rho1[0]["feature_dim"]

    def test_aug_overlap_sizes(self, ablation_results):
        records, _ = ablation_results
        # aug_overlap_size at rho=0 should equal 1 (just attr 0)
        for r in [r for r in records if r["rho"] == 0.0]:
            assert r["aug_overlap_size"] == 1

    def test_ca_abstains_at_rho_0(self, ablation_results):
        records, _ = ablation_results
        # CA with rho=0 should output exactly 0.5 (logit=0, no training params)
        ca_rho0 = [r for r in records if r["model"] == "closure_aware" and r["rho"] == 0.0]
        for r in ca_rho0:
            # n_epochs=0 means training was skipped (no parameters)
            assert r["n_epochs"] == 0

    def test_accuracy_in_range(self, ablation_results):
        records, _ = ablation_results
        for r in records:
            assert 0.0 <= r["test_accuracy"] <= 1.0

    def test_single_architecture_flag(self, tmp_path):
        ablation_main(
            ["--mini", "--architectures", "mlp", "closure_aware", "--output-dir", str(tmp_path)]
        )
        data = json.loads(list(tmp_path.glob("e1_ablation_*.json"))[0].read_text())
        models = {r["model"] for r in data["results"]}
        assert models == {"mlp", "closure_aware"}


# ---------------------------------------------------------------------------
# Analysis: plot is generated and has correct structure
# ---------------------------------------------------------------------------


class TestPlotAblation:
    def test_figure_created(self, ablation_results, tmp_path):
        from analysis.plot_ablation import plot

        records, _ = ablation_results
        out = tmp_path / "ablation.pdf"
        fig = plot(records, out)
        assert fig is not None
        assert out.exists() and out.stat().st_size > 0

    def test_figure_has_two_panels(self, ablation_results, tmp_path):
        from analysis.plot_ablation import plot

        records, _ = ablation_results
        out = tmp_path / "ablation_panels.pdf"
        fig = plot(records, out)
        assert len(fig.axes) == 2

    def test_latex_table_structure(self, ablation_results):
        from analysis.plot_ablation import latex_table

        records, _ = ablation_results
        tex = latex_table(records)
        assert r"\begin{table}" in tex
        assert "tab:ablation" in tex
        assert r"\toprule" in tex

    def test_latex_contains_rho_columns(self, ablation_results):
        from analysis.plot_ablation import latex_table

        records, _ = ablation_results
        tex = latex_table(records)
        assert "ρ=0.00" in tex
        assert "ρ=1.00" in tex

    def test_cli_runs(self, ablation_results, tmp_path):
        from analysis.plot_ablation import main as plot_main

        records, _ = ablation_results
        # Write records to a temp JSON file so CLI can load it
        import json

        src = tmp_path / "e1_ablation_test.json"
        src.write_text(json.dumps({"meta": {}, "results": records}))
        out = tmp_path / "ablation.pdf"
        plot_main(["--input", str(src), "--output", str(out)])
        assert out.exists()

    def test_cli_latex_flag(self, ablation_results, tmp_path, capsys):
        import json

        from analysis.plot_ablation import main as plot_main

        records, _ = ablation_results
        src = tmp_path / "e1_ablation_test2.json"
        src.write_text(json.dumps({"meta": {}, "results": records}))
        out = tmp_path / "ablation2.pdf"
        plot_main(["--input", str(src), "--output", str(out), "--latex"])
        assert "tab:ablation" in capsys.readouterr().out
