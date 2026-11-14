"""
tests/test_m4_calibration.py
============================
Automated tests for NEBULENS Milestone M4 (Calibration):
- M4.1: Frozen evaluation design, temporal splits, spatial blocks, leakage prevention,
        paired observation partitioning, promotion thresholds, and required output schema.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"
DESIGN_REPORT = PROJECT_ROOT / "docs" / "reports" / "m4_evaluation_design.md"
GRID_PATH = PROJECT_ROOT / "data" / "boundaries" / "grids" / "grid_10km.parquet"
PAIRS_PATH = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_4_aqua_viirs_pairs.parquet"


class TestM41EvaluationDesign:
    """Tests for Milestone M4.1: Frozen evaluation design and split integrity."""

    @pytest.fixture
    def config(self):
        assert CONFIG_PATH.is_file(), f"Config file missing: {CONFIG_PATH}"
        with open(CONFIG_PATH) as f:
            return json.load(f)

    def test_config_file_and_version(self, config):
        assert config["config_version"] == "v1.0.0-frozen"
        assert config["random_seed"] == 42
        assert DESIGN_REPORT.is_file(), f"Evaluation design report missing: {DESIGN_REPORT}"

    def test_temporal_splits_disjoint_and_complete(self, config):
        splits = config["temporal_splits"]
        train_start = pd.to_datetime(splits["train"]["start"])
        train_end = pd.to_datetime(splits["train"]["end"])
        val_start = pd.to_datetime(splits["validation"]["start"])
        val_end = pd.to_datetime(splits["validation"]["end"])
        test_start = pd.to_datetime(splits["test"]["start"])
        test_end = pd.to_datetime(splits["test"]["end"])

        # Check strict ordering and zero overlap
        assert train_start < train_end < val_start < val_end < test_start < test_end
        assert (val_start - train_end).days == 1
        assert (test_start - val_end).days == 1
        assert test_end == pd.to_datetime("2026-08-31")

    def test_spatial_blocks_coverage_and_independence(self, config):
        import geopandas as gpd
        g10 = gpd.read_parquet(GRID_PATH)
        assert len(g10) == 1460

        c = g10.geometry.centroid
        xs = c.x
        ys = c.y

        blocks = []
        for x, y in zip(xs, ys):
            if y < 1220000:
                blocks.append("Block_5_South")
            elif y >= 1450000:
                if x < 7550000:
                    blocks.append("Block_2_NorthWest")
                else:
                    blocks.append("Block_1_NorthEast")
            else:
                if x < 7550000:
                    blocks.append("Block_4_CentralWest")
                else:
                    blocks.append("Block_3_CentralEast_Delta")

        g10["spatial_block"] = blocks
        block_counts = g10["spatial_block"].value_counts().to_dict()

        for b_name, b_info in config["spatial_blocks"]["blocks"].items():
            assert b_name in block_counts
            assert block_counts[b_name] == b_info["expected_cells"]

        assert sum(block_counts.values()) == 1460

    def test_paired_observations_partition_integrity(self, config):
        pairs = pd.read_parquet(PAIRS_PATH)
        assert len(pairs) == 1809  # Verified baseline pairs

        train_end = config["temporal_splits"]["train"]["end"]
        val_start = config["temporal_splits"]["validation"]["start"]
        val_end = config["temporal_splits"]["validation"]["end"]
        test_start = config["temporal_splits"]["test"]["start"]

        p_train = pairs[pairs["date"] <= train_end]
        p_val = pairs[(pairs["date"] >= val_start) & (pairs["date"] <= val_end)]
        p_test = pairs[pairs["date"] >= test_start]

        assert len(p_train) + len(p_val) + len(p_test) == len(pairs)
        assert len(p_train) >= 500
        assert len(p_val) >= 500
        assert len(p_test) >= 700

        # Verify no pair overlap across sets
        assert set(p_train["aqua_id"]).isdisjoint(set(p_val["aqua_id"]))
        assert set(p_train["aqua_id"]).isdisjoint(set(p_test["aqua_id"]))
        assert set(p_val["aqua_id"]).isdisjoint(set(p_test["aqua_id"]))

    def test_required_output_fields_from_methodology(self, config):
        expected_five = [
            "observed_modis",
            "observed_viirs",
            "expected_sensor_specific_value",
            "harmonised_estimate",
            "uncertainty"
        ]
        assert config["required_output_fields"] == expected_five

    def test_promotion_rules_explicitly_defined(self, config):
        rules = config["promotion_rules"]
        assert "simplicity_rule" in rules
        assert rules["hurdle_zero_misfit_threshold_pct"] == 2.0
        assert rules["hurdle_min_mae_improvement_pct"] == 3.0
        assert rules["gradient_boosting_min_mae_improvement_pct"] == 5.0
        assert rules["sensor_transition_max_jump_increase_pct"] == 0.0


class TestM42ThroughM48Outputs:
    """Tests for Milestones M4.2 through M4.8 outputs and validation."""

    def test_m4_2_robust_baseline(self):
        metrics_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_2_baseline_metrics.csv"
        params_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_2_baseline_parameters.json"
        assert metrics_path.is_file(), "Baseline metrics CSV missing"
        assert params_path.is_file(), "Baseline parameters JSON missing"

        df = pd.read_csv(metrics_path)
        assert len(df) >= 3
        # Baseline should show strong positive correlation
        train_row = df[df["split"].str.lower() == "train"].iloc[0]
        assert train_row["count_pearson_r"] > 0.85

    def test_m4_3_negative_binomial_glm(self):
        metrics_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_metrics.csv"
        coeffs_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json"
        assert metrics_path.is_file(), "NB metrics CSV missing"
        assert coeffs_path.is_file(), "NB coefficients JSON missing"

        with open(coeffs_path) as f:
            d = json.load(f)
        assert d["converged"] is True
        assert "log_modis1" in d["coefficients"]
        assert d["coefficients"]["log_modis1"] > 0

    def test_m4_4_hurdle_decision(self):
        metrics_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_4_hurdle_metrics.csv"
        assert metrics_path.is_file(), "Hurdle metrics CSV missing"
        df = pd.read_csv(metrics_path)
        assert len(df) >= 3

    def test_m4_5_frp_calibration(self):
        pair_model_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_5_frp_pair_model.json"
        cell_model_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_5_frp_cell_model.json"
        assert pair_model_path.is_file(), "FRP pair model missing"
        assert cell_model_path.is_file(), "FRP cell model missing"

        with open(pair_model_path) as f:
            p = json.load(f)
        assert "scan_aqua" in p["coefficients"]

    def test_m4_6_model_selection_decision(self):
        decision_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_6_model_comparison_decision.json"
        assert decision_path.is_file(), "Model decision JSON missing"
        with open(decision_path) as f:
            dec = json.load(f)
        assert dec["selected_model"] == "Step_B_Negative_Binomial_GLM"
        assert "simplicity rule" in dec["rationale"].lower()

    def test_m4_7_bootstrap_uncertainty(self):
        models_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_7_bootstrap_models.json"
        cov_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_7_bootstrap_coverage.csv"
        unc_parquet = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_7_bootstrap_uncertainty.parquet"

        assert models_path.is_file(), "Bootstrap models JSON missing"
        assert cov_path.is_file(), "Bootstrap coverage CSV missing"
        assert unc_parquet.is_file(), "Bootstrap uncertainty parquet missing"

        with open(models_path) as f:
            m = json.load(f)
        assert m["n_replicates"] == 100
        assert len(m["count_params_all"]) == 100

        cov_df = pd.read_csv(cov_path)
        all_row = cov_df[cov_df["subset"].str.contains("All")].iloc[0]
        # 80% nominal interval should achieve at least 80% coverage
        assert all_row["cov_80_actual"] >= 80.0
        assert all_row["cov_95_actual"] >= 90.0

    def test_m4_8_harmonised_daily_dataset(self):
        summary_path = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_8_harmonised_summary.json"
        assert summary_path.is_file(), "Harmonised summary JSON missing"

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["n_grid_cells"] == 1460
        assert summary["total_cell_days"] == 13775100
        assert len(summary["yearly_breakdown"]) == 27

        # Check year=2020 partition
        p2020 = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily" / "year=2020" / "harmonised_daily_2020.parquet"
        assert p2020.is_file(), "2020 harmonised parquet missing"
        df2020 = pd.read_parquet(p2020)

        # Check exact required five fields
        required_five = [
            "observed_modis",
            "observed_viirs",
            "expected_sensor_specific_value",
            "harmonised_estimate",
            "uncertainty"
        ]
        for field in required_five:
            assert field in df2020.columns, f"Required field {field} missing from harmonised_daily schema"

        # Check provenance
        assert (df2020["h_source"] == "observed_viirs").all()
        assert (df2020["model_version"] == "m4_step_b_nb_glm_v1.0.0").all()

        # Check pre-VIIRS partition (e.g. 2005)
        p2005 = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily" / "year=2005" / "harmonised_daily_2005.parquet"
        assert p2005.is_file(), "2005 harmonised parquet missing"
        df2005 = pd.read_parquet(p2005)
        assert (df2005["h_source"] == "calibrated_from_modis").all()
