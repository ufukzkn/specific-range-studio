"""Frozen, common-input experiment data and portable artifact integrity."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["altitude", "gross_weight", "drag_index", "mach"]
KEYS = ["engine_type", *FEATURES]
CURVE_KEYS = KEYS[:-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info(root: Path) -> dict:
    def run(*args):
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        except OSError:
            # A USB runtime export need not include Git or a .git checkout.
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--porcelain"),
            "diff_sha256": hashlib.sha256((run("diff", "HEAD") or "").encode()).hexdigest(),
            "python_file_sha256": {str(p.relative_to(root).as_posix()): sha256(p)
                for folder in [root / "src", root / "scripts"] for p in sorted(folder.rglob("*.py"))}}


def prepare_partitions(frame: pd.DataFrame, seed: int = 42):
    """Quarantine conflicting labels; keep curve ends in train and hold out interior nodes.

    This evaluates unseen Mach nodes on known curves, not unseen flights or curves.
    Every identical condition occurs in at most one partition.
    """
    frame = frame.copy().reset_index(drop=True)
    frame["source_row_id"] = np.arange(len(frame))
    numeric = FEATURES + ["specific_range"]
    good = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1) & (frame.specific_range > 0)
    rejected = frame.loc[~good].copy()
    rejected["reason"] = "nonfinite_input_or_nonpositive_target"
    clean = frame.loc[good].copy()
    conflicting = clean.groupby(KEYS, dropna=False).specific_range.transform("nunique") > 1
    conflicts = clean.loc[conflicting].copy()
    conflicts["reason"] = "conflicting_targets_for_same_condition"
    rejected = pd.concat([rejected, conflicts], ignore_index=True)
    clean = clean.loc[~conflicting].sort_values(KEYS + ["source_row_id"])
    duplicate_count = int(clean.duplicated(KEYS).sum())
    clean = clean.drop_duplicates(KEYS).reset_index(drop=True)
    clean["row_id"] = np.arange(len(clean))
    pinned = []
    for _, group in clean.groupby(CURVE_KEYS, sort=True):
        ordered = group.sort_values("mach")
        pinned.extend(ordered.index if len(ordered) < 4 else [ordered.index[0], ordered.index[-1]])
    interior = clean.drop(index=pinned)
    if len(interior) < 10:
        raise ValueError("At least ten interior nodes are needed for a held-out experiment.")
    train_inner, rest = train_test_split(interior, test_size=0.30, random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, random_state=seed)
    train = pd.concat([clean.loc[pinned], train_inner]).sort_values("row_id")
    partitions = {"train": train, "valid": valid.sort_values("row_id"), "test": test.sort_values("row_id")}
    audit = {"input_rows": len(frame), "invalid_rows": int((~good).sum()),
             "conflicting_rows": len(conflicts), "duplicate_rows_removed": duplicate_count,
             "unique_conditions": len(clean), "boundary_nodes_pinned_to_train": len(pinned),
             "partition_rows": {k: len(v) for k, v in partitions.items()},
             "split_protocol": "known-curve interior Mach holdout; endpoints in train; seed-fixed; no condition overlap",
             "numerical_features": FEATURES, "categorical_features": ["engine_type"],
             "excluded_predictor": "fuel_flow (retained only as context; target-linked in current tables)",
             "specific_range_unit": "source numeric unit; workbook header NM is insufficient to verify NM/lb",
             "source_provenance": "user-provided digitized tables; independent ground truth not established"}
    return partitions, rejected, audit


def write_manifest(bundle: Path, metadata: dict):
    files = {str(p.relative_to(bundle).as_posix()): sha256(p)
             for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    payload = {"schema_version": 1, **metadata, "files": files}
    (bundle / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_bundle(bundle: Path) -> dict:
    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("files"):
        raise ValueError("Invalid bundle manifest")
    for relative, expected in payload["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Bundle integrity failed: {relative}")
    required = ["train.csv", "valid.csv", "test.csv", "xgboost/model.json", "xgboost/config.json",
                "xgboost/preprocessor.joblib", "ft_transformer/model.pt", "ft_transformer/preprocessor.joblib"]
    if any(name not in payload["files"] for name in required):
        raise ValueError("Incomplete bundle: partitions and both model artifacts are required")
    return payload
