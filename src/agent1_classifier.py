"""Agent 1: ResNet18 transfer-learning classifier for benign/malignant triage."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODEL_PATH = ROOT / "models" / "classifier.pt"

LABELS = ["benign", "malignant"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LesionDataset(Dataset):
    def __init__(self, csv_path: Path, train: bool):
        self.df = pd.read_csv(csv_path)
        self.transform = get_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = LABEL_TO_IDX[row["label"]]
        return image, label


def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = df["label"].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = df["label"].map(class_weights).values
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    report = classification_report(all_labels, all_preds, target_names=LABELS, output_dict=True, zero_division=0)
    malignant_recall = report["malignant"]["recall"]
    return avg_loss, malignant_recall, all_labels, all_preds


def train(epochs: int, batch_size: int, lr: float) -> None:
    device = get_device()
    print(f"device: {device}")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    train_ds = LesionDataset(SPLITS_DIR / "train.csv", train=True)
    val_ds = LesionDataset(SPLITS_DIR / "val.csv", train=False)

    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_malignant_recall = 0.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        val_loss, malignant_recall, val_labels, val_preds = evaluate(model, val_loader, device, criterion)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  malignant_recall={malignant_recall:.3f}")

        if malignant_recall >= best_malignant_recall:
            best_malignant_recall = malignant_recall
            torch.save({"model_state_dict": model.state_dict(), "labels": LABELS}, MODEL_PATH)
            print(f"  saved new best model (malignant_recall={malignant_recall:.3f}) -> {MODEL_PATH}")

    print("\nfinal val classification report (best checkpoint may differ from last epoch):")
    print(confusion_matrix(val_labels, val_preds))
    print(classification_report(val_labels, val_preds, target_names=LABELS, zero_division=0))


def load_model(device: torch.device | None = None) -> nn.Module:
    device = device or get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_threshold() -> float:
    """Decision threshold on p(malignant), tuned on validation by src/tune.py.

    Defaults to 0.5 (plain argmax) when no tuned threshold has been saved.
    """
    path = MODEL_PATH.parent / "threshold.json"
    if not path.exists():
        return 0.5
    try:
        return float(json.loads(path.read_text())["threshold"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.5


def predict(model: nn.Module, image_path: str, device: torch.device,
            threshold: float | None = None) -> dict:
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().tolist()
    class_probabilities = dict(zip(LABELS, probs))

    # Argmax is only correct at threshold 0.5. With an imbalanced, recall-sensitive
    # task the operating point is tuned, so decide on p(malignant) directly.
    thr = load_threshold() if threshold is None else threshold
    p_malignant = class_probabilities["malignant"]
    predicted_label = "malignant" if p_malignant >= thr else "benign"
    return {
        "predicted_label": predicted_label,
        "confidence": class_probabilities[predicted_label],
        "class_probabilities": class_probabilities,
        "threshold": thr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
