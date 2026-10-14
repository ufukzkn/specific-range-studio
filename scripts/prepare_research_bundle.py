"""Train a common-input comparison on the workstation; copy the bundle to Jetson."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.research_bundle import FEATURES, git_info, prepare_partitions, sha256, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_bundle_portable")
    parser.add_argument("--ft-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--label", default="research_candidate", help="Use functional_check for short test training")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; use a new directory for each training run")
    import torch
    import joblib
    import pandas as pd
    from src.data.load_data import load_combined_dataset
    from src.data.preprocess import category_cardinalities, fit_preprocessor, transform_split
    from src.models.ft_transformer import FTTransformerTrainer
    from src.models.xgboost_baseline import XGBoostRegressorWrapper
    from src.utils.config import DataConfig, FTTransformerConfig, XGBoostConfig
    from src.evaluation.heldout_reports import write_heldout_reports
    if args.ft_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; no CPU fallback")
    torch.set_num_threads(args.threads)
    config = DataConfig(numerical_features=tuple(FEATURES))
    frame = pd.read_csv(args.dataset) if args.dataset else load_combined_dataset(config)
    parts, rejected, audit = prepare_partitions(frame, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_csv(args.output / f"{name}.csv", index=False)
    rejected.to_csv(args.output / "quarantine.csv", index=False)
    (args.output / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Re-read frozen values so training and deployment use the identical CSV representation.
    parts = {name: pd.read_csv(args.output / f"{name}.csv") for name in parts}
    state = fit_preprocessor(parts["train"], config)
    prepared = {name: transform_split(part, state) for name, part in parts.items()}
    packages = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "joblib", "openpyxl", "threadpoolctl", "psutil"]}
    (args.output / "requirements-portable.txt").write_text(
        "# Match preprocessing versions; install NVIDIA PyTorch separately for the target JetPack.\n" +
        "\n".join(f"{name}=={version}" for name, version in packages.items() if name != "torch") + "\n", encoding="utf-8")
    summaries = {}
    for key in ["xgboost", "ft_transformer"]:
        start = time.perf_counter()
        folder = args.output / key
        folder.mkdir()
        joblib.dump(state, folder / "preprocessor.joblib")
        if key == "xgboost":
            model = XGBoostRegressorWrapper(XGBoostConfig(random_state=args.seed, device="cpu"))
            model.model.set_params(n_jobs=args.threads)
            model.fit(prepared["train"], prepared["valid"])
            model.save(folder / "model.json", folder / "config.json")
            fit = None
        else:
            model = FTTransformerTrainer(len(FEATURES), category_cardinalities(state),
                       FTTransformerConfig(epochs=args.epochs, device=args.ft_device, random_state=args.seed))
            fit = asdict(model.fit(prepared["train"], prepared["valid"]))
            model.save_checkpoint(folder / "model.pt", extra_metadata={"fit_result": fit, "experiment_label": args.label})
        summary = {"validation": model.evaluate(prepared["valid"]), "test": model.evaluate(prepared["test"]),
                   "fit_result": fit, "training_seconds": time.perf_counter() - start,
                   "config": asdict(model.config)}
        summaries[key] = summary
        (folder / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(key, json.dumps(summary), flush=True)
        def predict_test(batch):
            prepared_batch = transform_split(batch, state)
            return (model.predict_from_prepared(prepared_batch) if key == "xgboost"
                    else model.predict(prepared_batch.X_num, prepared_batch.X_cat))
        write_heldout_reports(parts["test"], key, predict_test, folder / "reports")
    sources = [args.dataset] if args.dataset else [config.one_engine_path, config.two_engine_path]
    write_manifest(args.output, {"created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_label": args.label, "seed": args.seed, "data_audit": audit, "training": summaries,
        "sources": {p.name: sha256(p) for p in sources}, "packages": packages,
        "source_code": git_info(ROOT), "torch_threads": args.threads, "report_scope": "held-out test partition only"})
    print(f"Bundle ready: {args.output}", flush=True)


if __name__ == "__main__":
    main()
