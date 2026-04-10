from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import SplitConfig


@dataclass(slots=True)
class DatasetSplit:
    """Container for train/validation/test dataframe splits."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_dataframe(frame: pd.DataFrame, config: SplitConfig | None = None) -> DatasetSplit:
    """Split a dataframe into train/validation/test partitions."""

    config = config or SplitConfig()
    if not abs(config.train_size + config.valid_size + config.test_size - 1.0) < 1e-8:
        raise ValueError("train_size + valid_size + test_size must sum to 1.0")

    train_df, temp_df = train_test_split(
        frame,
        test_size=(1.0 - config.train_size),
        random_state=config.random_state,
        shuffle=True,
    )
    relative_valid = config.valid_size / (config.valid_size + config.test_size)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_valid),
        random_state=config.random_state,
        shuffle=True,
    )
    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        valid=valid_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
