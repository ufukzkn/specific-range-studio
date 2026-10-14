"""Rebuild test-only UI reports from saved models, then update bundle integrity hashes."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.research_bundle import git_info, verify_bundle, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/research_bundle_portable")
    args = parser.parse_args()
    metadata = verify_bundle(args.bundle)
    import torch
    torch.set_num_threads(1)
    import pandas as pd
    from src.inference.predictors import XGBoostPredictor, FTTransformerPredictor
    from src.evaluation.heldout_reports import write_heldout_reports
    test = pd.read_csv(args.bundle / "test.csv")
    for key, loader in [("xgboost", XGBoostPredictor), ("ft_transformer", FTTransformerPredictor)]:
        predictor = loader.from_artifacts(args.bundle / key)
        if key == "xgboost":
            predictor.model.model.set_params(device="cpu", n_jobs=1)
        write_heldout_reports(test, key, predictor.predict_many_from_frame, args.bundle / key / "reports")
        print(f"{key}: {len(test)} held-out report rows", flush=True)
    metadata.pop("files")
    metadata["report_scope"] = "held-out test partition only"
    metadata["report_source_code"] = git_info(Path(__file__).resolve().parents[1])
    write_manifest(args.bundle, metadata)


if __name__ == "__main__":
    main()
