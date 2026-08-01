"""U-Net, masked objectives, spatial CV, and overlap prediction."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Three-level U-Net preserved from the source notebook."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = DoubleConv(in_channels, 64), DoubleConv(64, 128), DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3, self.dec3 = nn.ConvTranspose2d(512, 256, 2, 2), DoubleConv(512, 256)
        self.up2, self.dec2 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(256, 128)
        self.up1, self.dec1 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        d3 = self.dec3(torch.cat([self.up3(self.bottleneck(self.pool(e3))), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        return F.relu(self.out(self.dec1(torch.cat([self.up1(d2), e1], 1))))


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor):
    mask = torch.isfinite(pred) & torch.isfinite(target)
    return None if not mask.any() else F.mse_loss(pred[mask], target[mask])


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(pred[..., 1:] - pred[..., :-1]).mean()
        + torch.abs(pred[..., 1:, :] - pred[..., :-1, :]).mean()
    )


class SupervisedFloodDataset(Dataset):
    """Patch dataset with spatial-fold membership inherited from patch center."""

    def __init__(self, X_full: np.ndarray, y_full: np.ndarray, patch_size: int = 64,
                 stride: int | None = None, fold_map: np.ndarray | None = None,
                 val_fold: int | None = None, mode: str = "all",
                 drop_boundary: bool = False):
        self.X = np.asarray(X_full, np.float32)
        self.y = np.asarray(y_full, np.float32)[None]
        self.patch_size, self.stride = patch_size, stride or patch_size
        if self.X.ndim != 3 or self.y.shape[1:] != self.X.shape[1:]:
            raise ValueError("X must be CHW and y must share its spatial shape.")
        if fold_map is not None and fold_map.shape != self.X.shape[1:]:
            raise ValueError("fold_map shape mismatch.")
        self.samples: list[tuple[int, int]] = []
        _, height, width = self.X.shape
        for i in range(0, height - patch_size + 1, self.stride):
            for j in range(0, width - patch_size + 1, self.stride):
                if not np.isfinite(self.y[:, i:i+patch_size, j:j+patch_size]).any():
                    continue
                if fold_map is None or val_fold is None or mode == "all":
                    self.samples.append((i, j)); continue
                patch = fold_map[i:i+patch_size, j:j+patch_size]
                unique = np.unique(patch)
                if drop_boundary and len(unique) != 1:
                    continue
                assigned = int(unique[0] if drop_boundary else patch[patch_size//2, patch_size//2])
                if (mode == "train" and assigned != val_fold) or (mode == "val" and assigned == val_fold):
                    self.samples.append((i, j))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        i, j = self.samples[index]; p = self.patch_size
        return torch.from_numpy(self.X[:, i:i+p, j:j+p]), torch.from_numpy(self.y[:, i:i+p, j:j+p])


def validate_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval(); predicted, observed = [], []
    with torch.no_grad():
        for X, y in loader:
            pred = model(X.to(device)); y = y.to(device)
            mask = torch.isfinite(pred) & torch.isfinite(y)
            if mask.any():
                predicted.append(pred[mask].cpu().numpy()); observed.append(y[mask].cpu().numpy())
    if not predicted:
        return np.nan, np.nan, np.nan
    yp, yt = np.concatenate(predicted), np.concatenate(observed)
    errors = yp - yt
    mae, rmse = np.mean(np.abs(errors)), np.sqrt(np.mean(errors ** 2))
    denominator = np.sum((yt - yt.mean()) ** 2)
    return float(mae), float(rmse), float(np.nan if denominator == 0 else 1 - np.sum(errors ** 2) / denominator)


def build_spatial_fold_map(y: np.ndarray, n_folds: int = 5, block_size: int = 64,
                           seed: int = 42) -> np.ndarray:
    """Greedily balance flood pixels across contiguous spatial blocks."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    flood = np.isfinite(y) & (y > 0); height, width = y.shape
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            count = int(flood[i:min(i+block_size, height), j:min(j+block_size, width)].sum())
            blocks.append((count, i, min(i+block_size, height), j, min(j+block_size, width)))
    blocks.sort(reverse=True)
    fold_map = np.full((height, width), -1, np.int32); counts = [0] * n_folds
    for count, r0, r1, c0, c1 in blocks:
        fold = int(np.argmin(counts)); fold_map[r0:r1, c0:c1] = fold; counts[fold] += count
    if (fold_map < 0).any():
        raise ValueError("Spatial fold map contains unassigned pixels.")
    print("Flood pixels per fold:", counts)
    return fold_map


def run_spatial_cv_experiment(experiment_name: str, X: np.ndarray, y: np.ndarray,
                              fold_map: np.ndarray, config: dict[str, Any]):
    """Train a fresh U-Net per spatial fold; return metrics and loss histories."""
    if X.shape[1:] != y.shape or fold_map.shape != y.shape:
        raise ValueError("Experiment arrays are spatially misaligned.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity.")
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    results, histories, best_states = [], {}, {}
    n_folds = int(config["num_folds"])
    for fold in range(n_folds):
        train_ds = SupervisedFloodDataset(X, y, config["patch_size"], config["stride"], fold_map, fold, "train")
        val_ds = SupervisedFloodDataset(X, y, config["patch_size"], config["stride"], fold_map, fold, "val")
        if not train_ds or not val_ds:
            raise ValueError(f"Fold {fold + 1} has an empty train or validation dataset.")
        generator = torch.Generator().manual_seed(int(config.get("random_seed", 42)) + fold)
        train_loader = DataLoader(train_ds, config["unet_batch_size"], shuffle=True, generator=generator)
        val_loader = DataLoader(val_ds, config["unet_batch_size"], shuffle=False)
        model = UNet(in_channels=X.shape[0]).to(device)  # reset every fold
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_rmse, best_metrics, best_state, stale = np.inf, (np.nan, np.nan), None, 0
        train_history, val_history = [], []
        try:
            for epoch in range(int(config["epochs"])):
                model.train(); losses = []
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
                    pred = model(xb); primary = masked_mse_loss(pred, yb)
                    if primary is None:
                        continue
                    loss = primary + float(config["smoothness_weight"]) * smoothness_loss(pred)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("NaN/non-finite training loss.")
                    loss.backward(); optimizer.step(); losses.append(float(loss.item()))
                if not losses:
                    raise ValueError("Epoch contains no batches with valid target pixels.")
                mae, rmse, r2 = validate_model(model, val_loader, device)
                train_history.append(float(np.mean(losses))); val_history.append(rmse)
                if np.isfinite(rmse) and rmse < best_rmse - float(config.get("min_delta", 1e-4)):
                    best_rmse, best_metrics = rmse, (mae, r2)
                    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), epoch + 1, 0
                else:
                    stale += 1
                if stale >= int(config["early_stopping_patience"]):
                    break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("CUDA out of memory. Lower unet_batch_size or patch_size.")
            raise
        if best_state is None:
            raise RuntimeError(f"Fold {fold + 1} produced no finite validation result.")
        output_dir = Path(config.get("model_output_dir", "models")); output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / f"{experiment_name}_fold_{fold+1}.pth")
        best_states[fold + 1] = {k: v.cpu() for k, v in best_state.items()}
        histories[fold + 1] = {"train_loss": train_history, "val_rmse": val_history}
        results.append({"experiment": experiment_name, "fold": fold + 1, "best_epoch": best_epoch,
                        "MAE": best_metrics[0], "RMSE": best_rmse, "R2": best_metrics[1]})
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(results), histories, best_states


def train_final_model(X: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> tuple[nn.Module, list[float]]:
    """Train on all valid patches for the configured epoch count."""
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ds = SupervisedFloodDataset(X, y, config["patch_size"], config["stride"], mode="all")
    if not ds:
        raise ValueError("No valid final-training patches.")
    loader = DataLoader(ds, config["unet_batch_size"], shuffle=True)
    model = UNet(X.shape[0]).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    try:
        for _ in range(int(config["epochs"])):
            model.train(); losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); pred = model(xb)
                main = masked_mse_loss(pred, yb)
                if main is None:
                    continue
                loss = main + float(config["smoothness_weight"]) * smoothness_loss(pred)
                if not torch.isfinite(loss):
                    raise FloatingPointError("NaN/non-finite final-training loss.")
                loss.backward(); optimizer.step(); losses.append(float(loss.item()))
            if not losses:
                raise ValueError("No valid batches in final training.")
            history.append(float(np.mean(losses)))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); print("CUDA out of memory. Lower unet_batch_size or patch_size."); raise
    return model, history


def predict_full_map_overlap(model: nn.Module, X_full: np.ndarray, patch_size: int = 64,
                             stride: int = 32, batch_size: int = 4, device: str = "cpu") -> np.ndarray:
    """Average overlapping predictions, including padded edge patches."""
    model.eval(); _, height, width = X_full.shape
    pad_h = max(patch_size-height, 0) if height <= patch_size else (stride-(height-patch_size) % stride) % stride
    pad_w = max(patch_size-width, 0) if width <= patch_size else (stride-(width-patch_size) % stride) % stride
    padded = np.pad(X_full, ((0, 0), (0, pad_h), (0, pad_w)))
    total = np.zeros(padded.shape[1:], np.float32); count = np.zeros_like(total)
    patches, positions = [], []
    def flush() -> None:
        if not patches:
            return
        batch = torch.from_numpy(np.stack(patches).astype(np.float32)).to(device)
        with torch.no_grad():
            output = model(batch).cpu().numpy()[:, 0]
        for pred, (i, j) in zip(output, positions):
            total[i:i+patch_size, j:j+patch_size] += pred; count[i:i+patch_size, j:j+patch_size] += 1
        patches.clear(); positions.clear()
    for i in range(0, padded.shape[1]-patch_size+1, stride):
        for j in range(0, padded.shape[2]-patch_size+1, stride):
            patches.append(padded[:, i:i+patch_size, j:j+patch_size]); positions.append((i, j))
            if len(patches) == batch_size:
                flush()
    flush()
    return (total / np.maximum(count, 1))[:height, :width].astype(np.float32)
