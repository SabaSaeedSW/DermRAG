"""Hyperparameter and decision-threshold tuning for Agent 1.""" 

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

import agent1_classifier as a1

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
SWEEP_LOG = RESULTS_DIR / "sweep_results.jsonl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

MALIGNANT_IDX = a1.LABELS.index("malignant")


# shared

def val_probabilities(model: nn.Module, device: torch.device,
                      batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_malignant, y_true) over the validation split."""
    ds = a1.LesionDataset(SPLITS_DIR / "val.csv", train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            out = torch.softmax(model(images.to(device)), dim=1)
            probs.append(out[:, MALIGNANT_IDX].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics_at(p_malignant: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    y_pred = (p_malignant >= threshold).astype(int)
    y_bin = (y_true == MALIGNANT_IDX).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": float((y_pred == y_bin).mean()),
        "malignant_recall": float(recall_score(y_bin, y_pred, zero_division=0)),
        "malignant_precision": float(precision_score(y_bin, y_pred, zero_division=0)),
        "malignant_f1": float(f1_score(y_bin, y_pred, zero_division=0)),
    }


# threshold

def tune_threshold(min_recall: float) -> None:
    device = a1.get_device()
    print(f"device: {device}")
    model = a1.load_model(device)

    print("scoring validation split...")
    p_mal, y_true = val_probabilities(model, device)
    y_bin = (y_true == MALIGNANT_IDX).astype(int)
    auc = roc_auc_score(y_bin, p_mal)
    print(f"validation ROC AUC: {auc:.3f}  "
          f"(unchanged by thresholding : it bounds what tuning can achieve)\n")

    grid = [metrics_at(p_mal, y_true, t) for t in np.arange(0.05, 0.96, 0.01)]

    default = metrics_at(p_mal, y_true, 0.5)
    best_f1 = max(grid, key=lambda m: m["malignant_f1"])
    # Highest-precision threshold that still meets the recall floor we care about.
    meeting = [m for m in grid if m["malignant_recall"] >= min_recall]
    best_safe = max(meeting, key=lambda m: m["malignant_precision"]) if meeting else None

    def show(name: str, m: dict) -> None:
        print(f"{name:<28s} thr={m['threshold']:.2f}  acc={m['accuracy']:.3f}  "
              f"recall={m['malignant_recall']:.3f}  prec={m['malignant_precision']:.3f}  "
              f"F1={m['malignant_f1']:.3f}")

    print("=== threshold options (validation split) ===")
    show("current default (0.50)", default)
    show("best F1", best_f1)
    if best_safe:
        show(f"best precision @ recall>={min_recall:.2f}", best_safe)
    else:
        print(f"no threshold reaches {min_recall:.0%} malignant recall")

    chosen = best_safe or best_f1
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps({
        "threshold": chosen["threshold"],
        "selected_on": "validation",
        "criterion": (f"max precision subject to malignant recall >= {min_recall}"
                      if best_safe else "max malignant F1"),
        "validation_metrics": chosen,
        "validation_auc": float(auc),
        "saved_at": datetime.now().isoformat(),
    }, indent=2))
    print(f"\nsaved chosen threshold ({chosen['threshold']:.2f}) -> {THRESHOLD_PATH}")
    print("re-run `python3 src/evaluate.py` after regenerating results to see test-set effect.")


# sweep

def build_model(arch: str) -> nn.Module:
    """ResNet18 or EfficientNet-B0, both pretrained, with a fresh binary head."""
    from torchvision import models

    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, len(a1.LABELS))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(a1.LABELS))
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m


def train_one(config: dict, device: torch.device) -> dict:
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    train_ds = a1.LesionDataset(SPLITS_DIR / "train.csv", train=True)

    # Two ways to handle the nv-dominant imbalance: resample, or reweight the loss.
    if config["balance"] == "sampler":
        loader = DataLoader(train_ds, batch_size=config["batch_size"],
                            sampler=a1.make_weighted_sampler(train_df), num_workers=4)
        criterion = nn.CrossEntropyLoss()
    else:
        counts = train_df["label"].value_counts()
        weights = torch.tensor(
            [len(train_df) / (2 * counts[label]) for label in a1.LABELS],
            dtype=torch.float, device=device,
        )
        loader = DataLoader(train_ds, batch_size=config["batch_size"],
                            shuffle=True, num_workers=4)
        criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(config["arch"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best = {"val_auc": 0.0}
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        p_mal, y_true = val_probabilities(model, device)
        y_bin = (y_true == MALIGNANT_IDX).astype(int)
        auc = float(roc_auc_score(y_bin, p_mal))
        print(f"    epoch {epoch}/{config['epochs']}  "
              f"train_loss={running / len(loader.dataset):.4f}  val_auc={auc:.4f}")

        # Select on AUC: threshold-independent, and unlike raw recall it cannot be
        # gamed by collapsing to the positive class.
        if auc > best["val_auc"]:
            best = {"val_auc": auc, "epoch": epoch,
                    "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "p_mal": p_mal, "y_true": y_true}

    at_half = metrics_at(best["p_mal"], best["y_true"], 0.5)
    grid = [metrics_at(best["p_mal"], best["y_true"], t) for t in np.arange(0.02, 0.99, 0.01)]
    best_f1 = max(grid, key=lambda m: m["malignant_f1"])
    # Max-F1 is the wrong operating point for triage: on this data it roughly doubles
    # missed malignancies. Pick the highest-precision point that still clears the
    # recall floor, matching what tune_threshold does.
    meeting = [m for m in grid if m["malignant_recall"] >= config["min_recall"]]
    safe = max(meeting, key=lambda m: m["malignant_precision"]) if meeting else None
    return {**best, "val_at_0.5": at_half, "val_best_f1": best_f1, "val_safe": safe}


def run_sweep(epochs: int, quick: bool, min_recall: float) -> None:
    device = a1.get_device()
    print(f"device: {device}\n")

    if quick:
        grid = {"arch": ["resnet18"], "lr": [1e-4, 3e-4],
                "balance": ["sampler", "class_weight"], "batch_size": [32],
                "weight_decay": [1e-4]}
    else:
        # The quick sweep showed lr and the balance strategy barely matter (AUC spread
        # 0.002), so we don't re-explore them exhaustively. The untested axes that could
        # still move the needle are architecture and regularisation strength.
        grid = {"arch": ["resnet18", "efficientnet_b0"], "lr": [1e-4],
                "balance": ["class_weight"], "batch_size": [32],
                "weight_decay": [1e-4, 1e-2]}

    keys = list(grid)
    configs = [dict(zip(keys, values), epochs=epochs, min_recall=min_recall)
               for values in itertools.product(*(grid[k] for k in keys))]
    print(f"{len(configs)} configurations x {epochs} epochs\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = []

    for i, config in enumerate(configs, start=1):
        label = (f"{config['arch']} lr={config['lr']:g} {config['balance']} "
                 f"wd={config['weight_decay']:g}")
        print(f"[{i}/{len(configs)}] {label}")
        result = train_one(config, device)

        record = {
            "config": config,
            "best_epoch": result["epoch"],
            "val_auc": result["val_auc"],
            "val_at_0.5": result["val_at_0.5"],
            "val_best_f1": result["val_best_f1"],
            "val_safe": result["val_safe"],
        }
        leaderboard.append((result["val_auc"], config, result))
        with open(SWEEP_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  -> best val AUC {result['val_auc']:.4f} at epoch {result['epoch']}\n")

    leaderboard.sort(key=lambda x: -x[0])
    floor = configs[0]["min_recall"]
    print(f"=== leaderboard (validation ROC AUC; precision at recall>={floor:.2f}) ===")
    for auc, config, result in leaderboard:
        safe = result["val_safe"]
        prec = f"{safe['malignant_precision']:.3f}" if safe else "  n/a"
        print(f"  AUC={auc:.4f}  prec@floor={prec}  {config['arch']:16s} "
              f"lr={config['lr']:<7g} {config['balance']:13s} wd={config['weight_decay']:<6g}")

    best_auc, best_config, best_result = leaderboard[0]
    out = ROOT / "models" / "classifier_tuned.pt"
    torch.save({"model_state_dict": best_result["state_dict"], "labels": a1.LABELS,
                "arch": best_config["arch"], "config": best_config,
                "val_auc": best_auc}, out)

    chosen = best_result["val_safe"] or best_result["val_best_f1"]
    THRESHOLD_PATH.write_text(json.dumps({
        "threshold": chosen["threshold"],
        "selected_on": "validation",
        "criterion": (f"max precision subject to malignant recall >= {floor}"
                      if best_result["val_safe"] else
                      f"max malignant F1 (no threshold reached recall {floor})"),
        "validation_metrics": chosen,
        "validation_auc": best_auc,
        "saved_at": datetime.now().isoformat(),
    }, indent=2))

    print(f"\nsaved best model -> {out}")
    print(f"saved threshold  -> {THRESHOLD_PATH}")
    print(f"full log         -> {SWEEP_LOG}")
    print("\nthis did NOT overwrite models/classifier.pt : compare on test first, then")
    print("promote deliberately if the tuned model wins.")


def test_probabilities(model: nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    ds = a1.LesionDataset(SPLITS_DIR / "test.csv", train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            out = torch.softmax(model(images.to(device)), dim=1)
            probs.append(out[:, MALIGNANT_IDX].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def compare() -> None:
    """Final head-to-head on the held-out test split. Selection already happened on val."""
    device = a1.get_device()
    tuned_path = ROOT / "models" / "classifier_tuned.pt"
    if not tuned_path.exists():
        raise SystemExit("no tuned model yet : run  python3 src/tune.py sweep --quick")

    baseline = a1.load_model(device)
    ckpt = torch.load(tuned_path, map_location=device)
    tuned = build_model(ckpt["arch"])
    tuned.load_state_dict(ckpt["model_state_dict"])
    tuned.to(device).eval()

    thr = json.loads(THRESHOLD_PATH.read_text())["threshold"]
    pb, y = test_probabilities(baseline, device)
    pt, _ = test_probabilities(tuned, device)
    n_mal = int((y == MALIGNANT_IDX).sum())

    print(f"=== held-out TEST ({len(y)} images, {n_mal} malignant) ===")
    print("selection was done on validation; test is used once, here.\n")
    print(f"{'model':<20s} {'AUC':>6s} {'acc':>7s} {'recall':>7s} {'prec':>7s} "
          f"{'F1':>6s} {'missed':>7s}")
    for name, p, t in [("baseline @0.50", pb, 0.5), ("tuned    @0.50", pt, 0.5),
                       (f"tuned    @{thr:.2f}", pt, thr)]:
        auc = roc_auc_score((y == MALIGNANT_IDX).astype(int), p)
        m = metrics_at(p, y, t)
        missed = round(n_mal * (1 - m["malignant_recall"]))
        print(f"{name:<20s} {auc:6.3f} {m['accuracy']:7.3f} "
              f"{m['malignant_recall']:7.3f} {m['malignant_precision']:7.3f} "
              f"{m['malignant_f1']:6.3f} {missed:4d}/{n_mal}")

    # Isolate genuine model improvement from operating-point choice.
    print("\nprecision at matched malignant recall:")
    print(f"{'recall':>8s} {'baseline':>9s} {'tuned':>8s} {'delta':>8s}")
    for target in (0.95, 0.90, 0.887, 0.85, 0.80):
        def best_prec(p):
            ok = [metrics_at(p, y, t) for t in np.arange(0.01, 0.995, 0.005)]
            ok = [m for m in ok if m["malignant_recall"] >= target]
            return max(ok, key=lambda m: m["malignant_precision"]) if ok else None
        b, t_ = best_prec(pb), best_prec(pt)
        if not b or not t_:
            print(f"{target:8.3f} {'n/a':>9s} {'n/a':>8s}")
            continue
        d = t_["malignant_precision"] - b["malignant_precision"]
        print(f"{target:8.3f} {b['malignant_precision']:9.3f} "
              f"{t_['malignant_precision']:8.3f} {d:+8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["threshold", "sweep", "compare"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="small grid (4 configs)")
    parser.add_argument("--min-recall", type=float, default=0.90,
                        help="malignant recall floor for threshold selection")
    args = parser.parse_args()

    if args.mode == "threshold":
        tune_threshold(args.min_recall)
    elif args.mode == "compare":
        compare()
    else:
        run_sweep(args.epochs, args.quick, args.min_recall)
