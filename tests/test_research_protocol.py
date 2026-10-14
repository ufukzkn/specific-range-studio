import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.data.load_data import WorkbookSpec, read_workbook
from src.evaluation.research_bundle import FEATURES, KEYS, CURVE_KEYS, git_info, prepare_partitions, sha256, verify_bundle
from src.interpolation.specific_range import CubicSplineInterpolator, SpecificRangeInterpolationService


def table():
    rows = []
    for engine in ["one_engine", "two_engine"]:
        for altitude in [0.0, 5000.0]:
            for mach in np.linspace(0.2, 0.9, 30):
                rows.append(dict(engine_type=engine, altitude=altitude, gross_weight=30000.0,
                    drag_index=0.0, mach=mach, specific_range=0.2+mach*mach, fuel_flow=2000.0))
    return pd.DataFrame(rows)


class ResearchProtocolTests(unittest.TestCase):
    def test_usb_source_record_works_without_git_installed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "src/example.py").write_text("value = 1\n")
            with patch("src.evaluation.research_bundle.subprocess.run", side_effect=FileNotFoundError("git")):
                info = git_info(root)
            self.assertIsNone(info["commit"])
            self.assertIsNone(info["status"])
            self.assertIn("src/example.py", info["python_file_sha256"])

    def test_text_sea_level_survives_workbook_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.xlsx"
            pd.DataFrame({"Altitude (ft)": ["Sea Level", "sea level", "0"],
                          "Mach Number (Ma)": [0.2, 0.3, 0.4]}).to_excel(path, sheet_name="Sea Level", index=False)
            loaded = read_workbook(WorkbookSpec(path, "one_engine"))
            np.testing.assert_array_equal(loaded.altitude, [0.0, 0.0, 0.0])

    def test_natural_cubic_matches_independent_scipy_reference(self):
        x = np.array([0.1, 0.23, 0.51, 0.78, 0.9])
        y = np.array([1.0, 1.4, 1.9, 1.7, 1.5])
        spline = CubicSplineInterpolator(x, y)
        queries = np.linspace(x[0], x[-1], 70)
        np.testing.assert_allclose([spline.evaluate(q) for q in queries], CubicSpline(x, y, bc_type="natural")(queries), atol=1e-12)

    def test_no_duplicate_or_condition_overlap_between_partitions(self):
        source = pd.concat([table(), table().iloc[[0]]], ignore_index=True)
        parts, _, audit = prepare_partitions(source)
        self.assertEqual(audit["duplicate_rows_removed"], 1)
        sets = {name: set(map(tuple, part[KEYS].to_numpy())) for name, part in parts.items()}
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            self.assertFalse(sets[a] & sets[b])

    def test_conflicting_conditions_quarantined_in_full(self):
        source = table()
        changed = source.iloc[[10]].copy()
        changed.specific_range += 0.4
        parts, rejected, audit = prepare_partitions(pd.concat([source, changed], ignore_index=True))
        self.assertEqual(audit["conflicting_rows"], 2)
        self.assertEqual(len(rejected), 2)
        combined = pd.concat(list(parts.values()))
        self.assertFalse(((combined[KEYS] == changed[KEYS].iloc[0]).all(axis=1)).any())

    def test_holdout_is_interior_to_its_training_curve(self):
        parts, _, _ = prepare_partitions(table())
        ranges = parts["train"].groupby(CURVE_KEYS).mach.agg(["min", "max"])
        for _, row in pd.concat([parts["valid"], parts["test"]]).iterrows():
            bounds = ranges.loc[tuple(row[CURVE_KEYS])]
            self.assertTrue(bounds["min"] < row.mach < bounds["max"])

    def test_split_reproducible(self):
        a, _, _ = prepare_partitions(table(), 7)
        b, _, _ = prepare_partitions(table(), 7)
        for name in a:
            pd.testing.assert_frame_equal(a[name], b[name])

    def test_interpolation_uses_only_supplied_reference(self):
        parts, _, _ = prepare_partitions(table())
        train = parts["train"]
        query = parts["test"].iloc[[0]].copy()
        service = SpecificRangeInterpolationService.from_frame(train)
        value = service.predict_many_from_frame(query)
        query.specific_range = 99999
        query.fuel_flow = -99999
        np.testing.assert_array_equal(value, service.predict_many_from_frame(query))
        query.engine_type = "missing_engine"
        with self.assertRaisesRegex(ValueError, "absent"):
            service.predict_many_from_frame(query)

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train.csv"
            path.write_text("original")
            (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {"train.csv": sha256(path)}}))
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_bundle(root)

    def test_manifest_rejects_incomplete_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train.csv"
            path.write_text("original")
            (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {"train.csv": sha256(path)}}))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_bundle(root)


if __name__ == "__main__":
    unittest.main()
