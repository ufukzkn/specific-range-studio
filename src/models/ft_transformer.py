from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.evaluation.metrics import regression_metrics
from src.utils.config import FTTransformerConfig
from src.utils.device import resolve_torch_device
from src.utils.seed import set_global_seed

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the FT-Transformer model. Install requirements.txt first."
    ) from exc


class TabularDataset(Dataset):
    """Simple torch dataset for mixed numerical and categorical tabular data."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray) -> None:
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_num[index], self.x_cat[index], self.y[index]


class NumericalTokenizer(nn.Module):
    """Project each numerical feature into a token embedding."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerRegressor(nn.Module):
    """Practical FT-Transformer-style regressor for mixed tabular inputs."""

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        config: FTTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FTTransformerConfig()
        self.categorical_cardinalities = categorical_cardinalities
        d_model = self.config.d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.num_tokenizer = NumericalTokenizer(num_numeric_features, d_model)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in categorical_cardinalities]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num_tokens = self.num_tokenizer(x_num)
        cat_tokens = [embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)]
        if cat_tokens:
            cat_tokens_tensor = torch.stack(cat_tokens, dim=1)
        else:
            cat_tokens_tensor = torch.empty((x_num.size(0), 0, self.config.d_model), device=x_num.device)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        tokens = torch.cat([cls, num_tokens, cat_tokens_tensor], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(-1)


@dataclass(slots=True)
class FitResult:
    """Training summary for the FT-Transformer model."""

    best_validation_rmse: float
    epochs_trained: int


class FTTransformerTrainer:
    """Wrapper that handles training, early stopping, and evaluation."""

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        config: FTTransformerConfig | None = None,
    ) -> None:
        self.config = config or FTTransformerConfig()
        set_global_seed(self.config.random_state)
        self.config.device = resolve_torch_device(self.config.device)
        self.device = torch.device(self.config.device)
        self.model = FTTransformerRegressor(
            num_numeric_features=num_numeric_features,
            categorical_cardinalities=categorical_cardinalities,
            config=self.config,
        ).to(self.device)

    def _make_loader(self, split, shuffle: bool) -> DataLoader:
        dataset = TabularDataset(split.X_num, split.X_cat, split.y)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def fit(self, train_split, valid_split) -> FitResult:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()
        train_loader = self._make_loader(train_split, shuffle=True)
        valid_loader = self._make_loader(valid_split, shuffle=False)

        best_state = None
        best_rmse = float("inf")
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            for x_num, x_cat, y in train_loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(x_num, x_cat)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()

            valid_predictions = self.predict_split(valid_loader)
            metrics = regression_metrics(valid_split.y, valid_predictions)
            current_rmse = metrics["rmse"]
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return FitResult(best_validation_rmse=best_rmse, epochs_trained=epoch)

    @torch.no_grad()
    def predict_split(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        outputs: list[np.ndarray] = []
        for x_num, x_cat, _ in loader:
            predictions = self.model(x_num.to(self.device), x_cat.to(self.device))
            outputs.append(predictions.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    @torch.no_grad()
    def predict(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor_num = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        tensor_cat = torch.tensor(x_cat, dtype=torch.long, device=self.device)
        predictions = self.model(tensor_num, tensor_cat)
        return predictions.detach().cpu().numpy()

    def evaluate(self, split) -> dict[str, float]:
        predictions = self.predict(split.X_num, split.X_cat)
        return regression_metrics(split.y, predictions)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.model.parameters()))

    def save_checkpoint(self, path: Path, extra_metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "num_numeric_features": self.model.num_tokenizer.weight.shape[0],
                "categorical_cardinalities": self.model.categorical_cardinalities,
                "extra_metadata": extra_metadata or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Path, *, device: str = "cpu") -> "FTTransformerTrainer":
        checkpoint = torch.load(path, map_location=resolve_torch_device(device))
        config = FTTransformerConfig(**checkpoint["config"])
        config.device = resolve_torch_device(device)
        trainer = cls(
            num_numeric_features=int(checkpoint["num_numeric_features"]),
            categorical_cardinalities=list(checkpoint["categorical_cardinalities"]),
            config=config,
        )
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.to(trainer.device)
        trainer.model.eval()
        return trainer
