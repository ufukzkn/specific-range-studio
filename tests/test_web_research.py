"""Integration checks against a locally prepared real artifact bundle."""
import importlib.util
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/research_bundle_portable"


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "Prepare the real research bundle first")
class WebResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SPECIFIC_RANGE_BUNDLE"] = str(BUNDLE)
        spec = importlib.util.spec_from_file_location("research_server", ROOT / "scripts/web_app/server.py")
        cls.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.server)
        cls.client = cls.server.app.test_client()

    def test_research_page_renders_and_exposes_new_benchmark_panel(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"edge-benchmark-status", response.data)
        self.assertNotIn(b'id="pi-benchmark-status"', response.data)

    def test_both_engine_categories_predict_without_fuel_input(self):
        frame = self.server._get_reference_df()
        for engine in ["one_engine", "two_engine"]:
            row = frame.loc[frame.engine_type == engine].iloc[0]
            query = {k: float(row[k]) for k in ["altitude", "gross_weight", "drag_index", "mach"]}
            query["engine_type"] = engine
            a = self.client.post("/api/predict", json=query)
            query["fuel_flow"] = 999999
            b = self.client.post("/api/predict", json=query)
            self.assertEqual(a.status_code, 200)
            self.assertEqual(b.status_code, 200)
            for key in ["interpolation", "xgboost", "ft_transformer"]:
                self.assertIn(key, a.json)
                self.assertAlmostEqual(a.json[key], b.json[key], places=10)

    def test_interpolation_summary_has_no_invented_accuracy(self):
        metrics = self.server._load_model_summary("interpolation")["metrics"]
        self.assertNotIn("rmse", metrics)
        self.assertNotIn("r2", metrics)

    def test_legacy_training_and_pi_commands_cannot_mutate_research_profile(self):
        self.assertEqual(self.client.post("/api/setup/run", json={}).status_code, 409)
        self.assertEqual(self.client.post("/api/benchmark/pi/run", json={}).status_code, 409)

    def test_local_edge_results_are_identified(self):
        result = self.client.get("/api/benchmark/edge/latest")
        if result.status_code == 200:
            self.assertIn("verified_jetson", result.json)
            self.assertIn("experiment_label", result.json)
        else:
            self.assertEqual(result.status_code, 404)

    def test_ui_report_contains_only_frozen_test_ids(self):
        import pandas as pd
        expected = pd.read_csv(BUNDLE / "test.csv").row_id.tolist()
        for key in ["xgboost", "ft_transformer"]:
            path = BUNDLE / key / "reports" / f"{key}_row_level_comparison.csv"
            if not path.exists():
                self.skipTest("Refresh held-out reports first")
            rows = pd.read_csv(path)
            self.assertEqual(rows.row_id.tolist(), expected)
            response = self.client.get(f"/api/report/{key}/metrics")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["rows"], len(expected))


if __name__ == "__main__":
    unittest.main()
