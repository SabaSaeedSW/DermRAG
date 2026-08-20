"""Agent 2: CLIP embeddings + FAISS index for image-based retrieval over train+val."""

import os

# faiss and torch link conflicting OpenMP runtimes on macOS, which segfaults
# unless resolved before either is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
INDEX_DIR = ROOT / "index"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "index_metadata.csv"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def get_device() -> torch.device:
    # MPS has shown intermittent segfaults loading this CLIP checkpoint;
    # CPU is slower but reliable for a one-time embedding pass.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.to(device)
    model.eval()
    return model, preprocess


def embed_images(image_paths: list[str], model, preprocess, device: torch.device, batch_size: int = 64) -> np.ndarray:
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
            features = model.encode_image(batch_tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())
            print(f"  embedded {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end="\r")
    print()
    return np.concatenate(all_embeddings, axis=0).astype("float32")


def build_index() -> None:
    device = get_device()
    print(f"device: {device}")
    model, preprocess = load_clip(device)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    index_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"embedding {len(index_df)} images (train+val)...")
    embeddings = embed_images(index_df["image_path"].tolist(), model, preprocess, device)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    index_df[["image_id", "lesion_id", "dx", "label", "image_path"]].to_csv(METADATA_PATH, index=False)
    print(f"saved index ({index.ntotal} vectors, dim={dim}) -> {INDEX_PATH}")
    print(f"saved metadata -> {METADATA_PATH}")


class Retriever:
    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        self.model, self.preprocess = load_clip(self.device)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.metadata = pd.read_csv(METADATA_PATH)

    def retrieve(self, image_path: str, k: int = 5) -> list[dict]:
        with torch.no_grad():
            tensor = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            query_embedding = features.cpu().numpy().astype("float32")

        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.metadata.iloc[idx]
            results.append({
                "image_id": row["image_id"],
                "dx": row["dx"],
                "label": row["label"],
                "image_path": row["image_path"],
                "similarity": float(score),
            })
        return results


def eyeball_check(n_queries: int = 10, k: int = 5, stratified: bool = False) -> None:
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    if stratified:
        per_class = max(1, n_queries // test_df["dx"].nunique())
        sample = pd.concat([
            group.sample(n=min(per_class, len(group)), random_state=42)
            for _, group in test_df.groupby("dx")
        ], ignore_index=True)
    else:
        sample = test_df.sample(n=n_queries, random_state=42)

    retriever = Retriever()
    hits, total = 0, 0
    per_class_hits: dict[str, list[int]] = {}
    for _, row in sample.iterrows():
        results = retriever.retrieve(row["image_path"], k=k)
        neighbor_dx = [r["dx"] for r in results]
        matches = sum(1 for dx in neighbor_dx if dx == row["dx"])
        hits += matches
        total += k
        per_class_hits.setdefault(row["dx"], []).append(matches)
        print(f"query {row['image_id']} (dx={row['dx']}) -> neighbors dx={neighbor_dx}  ({matches}/{k} match)")

    print(f"\noverall precision@{k} on {len(sample)} sampled queries: {hits}/{total} = {hits/total:.1%}")
    print("\nper-class precision@{}:".format(k))
    for dx, match_list in sorted(per_class_hits.items()):
        class_hits = sum(match_list)
        class_total = len(match_list) * k
        print(f"  {dx}: {class_hits}/{class_total} = {class_hits/class_total:.1%}  (n={len(match_list)} queries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["build", "eyeball"], default="build")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--stratified", action="store_true")
    args = parser.parse_args()

    if args.action == "build":
        build_index()
    else:
        eyeball_check(n_queries=args.n_queries, k=args.k, stratified=args.stratified)
