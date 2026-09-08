"""Train one configured classifier experiment; no implicit dataset preparation."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from krishidrishti_ai.config import load_config
from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.models.classifier import build_model


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval(); total_loss = total = 0; predictions: list[int] = []; targets: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device); logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0); total += labels.size(0)
            predictions.extend(logits.argmax(1).cpu().tolist()); targets.extend(labels.cpu().tolist())
    macro_f1 = precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)[2]
    return {"loss": total_loss / total, "accuracy": sum(p == y for p, y in zip(predictions, targets)) / total, "macro_f1": float(macro_f1)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); train_config = config["training"]; set_seed(config["data"]["seed"], train_config["deterministic"])
    device = torch.device("cuda" if train_config["device"] == "auto" and torch.cuda.is_available() else "cpu" if train_config["device"] == "auto" else train_config["device"])
    root = Path(config["paths"]["processed_data_dir"])
    robust = train_config.get("robust_augmentation", False)
    train_set = ImageFolder(root / "train", build_transforms(config["data"]["image_size"], True, robust=robust)); val_set = ImageFolder(root / "val", build_transforms(config["data"]["image_size"], False))
    if train_set.classes != val_set.classes: raise ValueError("Train and validation class mappings differ")
    loader_args = {"batch_size": train_config["batch_size"], "num_workers": train_config["num_workers"], "pin_memory": device.type == "cuda"}
    generator = torch.Generator().manual_seed(config["data"]["seed"])
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args); val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = build_model(config["model"]["name"], len(train_set.classes), config["model"]["pretrained"]).to(device)
    weight = None
    if train_config["class_balance"] == "loss_weights":
        counts = np.bincount(train_set.targets, minlength=len(train_set.classes)); weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    elif train_config["class_balance"] != "none": raise ValueError("training.class_balance must be 'none' or 'loss_weights'")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=train_config["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["epochs"]) if train_config["scheduler"] == "cosine" else None
    checkpoint = Path(config["paths"]["checkpoint_path"]); checkpoint.parent.mkdir(parents=True, exist_ok=True); best = float("-inf"); stale = 0
    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True)
            criterion(model(images), labels).backward(); optimizer.step()
        metrics = validate(model, val_loader, criterion, device); selected = metrics[train_config["checkpoint_metric"]]
        print(f"epoch={epoch} val_loss={metrics['loss']:.4f} val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}")
        if selected > best:
            best, stale = selected, 0
            torch.save({"model_state_dict": model.state_dict(), "classes": train_set.classes, "model_name": config["model"]["name"], "config": config, "best_validation_metrics": metrics, "epoch": epoch}, checkpoint)
        else: stale += 1
        if scheduler: scheduler.step()
        if stale >= train_config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}; best {train_config['checkpoint_metric']}={best:.4f}"); break
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__": main()
