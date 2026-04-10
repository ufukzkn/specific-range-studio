from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_combined_dataset, save_processed_dataset
from src.utils.config import DataConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and clean the aircraft specific range workbooks.")
    parser.add_argument("--output", type=str, default=None, help="Optional custom CSV output path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_config = DataConfig()
    dataset = load_combined_dataset(data_config)
    output_path = save_processed_dataset(dataset, data_config)
    if args.output:
        dataset.to_csv(args.output, index=False)
        output_path = args.output
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {list(dataset.columns)}")
    print(f"Saved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
