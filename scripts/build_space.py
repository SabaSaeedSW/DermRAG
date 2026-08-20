"""Assemble a self-contained Hugging Face Space bundle in space/."""

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "space"

THUMB_PX = 160          # retrieval neighbours are shown small
EXAMPLE_PX = 600        # example queries are the input, keep them full size
JPEG_QUALITY = 82

# On Linux the default PyPI torch wheel bundles CUDA and is several GB, which exceeds
# the limits on both Streamlit Community Cloud and a Space build. Making the PyTorch
# CPU index primary forces the small CPU wheels; that index carries only torch
# packages, so everything else falls through to PyPI via the extra index.
SPACE_REQUIREMENTS = """\
--index-url https://download.pytorch.org/whl/cpu
--extra-index-url https://pypi.org/simple

torch
torchvision
streamlit
open_clip_torch
faiss-cpu
pandas
numpy
scikit-learn
pillow
google-genai
anthropic
"""

SPACE_DOCKERFILE = """\
FROM python:3.11-slim

# libgomp is required by faiss-cpu; libglib/libgl by pillow-simd style image ops.
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgomp1 libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \\
    PYTHONUNBUFFERED=1 \\
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY --chown=user requirements.txt .
# requirements.txt pins the PyTorch CPU index, so this pulls the small CPU wheels.
RUN pip install --no-cache-dir --upgrade pip \\
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \\
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
"""

SPACE_README = """\
---
title: DermRAG
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DermRAG

**Research and educational demo, not a diagnostic tool.** Trained on a single public
dataset, it makes frequent mistakes and must not inform decisions about anyone's health.

A three-agent pipeline for dermatoscopic lesion triage:

1. **Classifier** — fine-tuned ResNet18, benign vs malignant.
2. **Retriever** — CLIP ViT-B-32 embeddings + FAISS over 8,521 labelled reference cases.
3. **Reasoner** — an LLM writes a rationale grounded strictly in the outputs above.

Set `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY`) as a Space secret to enable the reasoner.
Without one, agents 1 and 2 still run.

Built on HAM10000. Inspired by *PathFinder* (RAIVN Lab, University of Washington, 2025).
"""


def rel_thumb(image_id: str) -> str:
    return f"assets/thumbs/{image_id}.jpg"


def rel_example(image_id: str) -> str:
    return f"assets/examples/{image_id}.jpg"


def write_thumbnails(df: pd.DataFrame, dest: Path, px: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        src = Path(row["image_path"])
        if not src.exists():
            continue
        out = dest / f"{row['image_id']}.jpg"
        if not out.exists():
            im = Image.open(src).convert("RGB")
            im.thumbnail((px, px))
            im.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
        written += 1
        if i % 1000 == 0:
            print(f"    {i}/{len(df)}")
    return written


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    (OUT / "models").mkdir(parents=True)
    (OUT / "index").mkdir(parents=True)
    (OUT / "data" / "splits").mkdir(parents=True)

    # code
    for name in ["agent1_classifier.py", "agent2_retriever.py", "agent3_reasoner.py"]:
        shutil.copy2(ROOT / "src" / name, OUT / "src" / name)
    shutil.copy2(ROOT / "app.py", OUT / "app.py")
    print("copied app + agent modules")

    # model + index
    shutil.copy2(ROOT / "models" / "classifier.pt", OUT / "models" / "classifier.pt")
    shutil.copy2(ROOT / "models" / "threshold.json", OUT / "models" / "threshold.json")
    shutil.copy2(ROOT / "index" / "faiss_index.bin", OUT / "index" / "faiss_index.bin")
    print("copied classifier, threshold, FAISS index")

    # reference thumbnails, with paths rewritten 
    index_df = pd.read_csv(ROOT / "index" / "index_metadata.csv")
    print(f"building {len(index_df)} reference thumbnails at {THUMB_PX}px...")
    n = write_thumbnails(index_df, OUT / "assets" / "thumbs", THUMB_PX)
    index_df["image_path"] = index_df["image_id"].map(rel_thumb)
    index_df.to_csv(OUT / "index" / "index_metadata.csv", index=False)
    print(f"  wrote {n} thumbnails, rewrote paths to assets/thumbs/")

    # one example query per diagnosis
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    examples = pd.concat([g.iloc[[0]] for _, g in test_df.groupby("dx")], ignore_index=True)
    write_thumbnails(examples, OUT / "assets" / "examples", EXAMPLE_PX)
    examples["image_path"] = examples["image_id"].map(rel_example)
    examples.to_csv(OUT / "data" / "splits" / "test.csv", index=False)
    print(f"  wrote {len(examples)} example images (one per diagnosis)")

    # pace metadata 
    (OUT / "requirements.txt").write_text(SPACE_REQUIREMENTS)
    (OUT / "README.md").write_text(SPACE_README)
    (OUT / "Dockerfile").write_text(SPACE_DOCKERFILE)
    # Streamlit writes its config and the transient query image at runtime; the
    # container filesystem is writable only under the app user's home and /app.
    (OUT / ".dockerignore").write_text(".git\n__pycache__/\n*.pyc\n")
    (OUT / ".gitattributes").write_text(
        "*.pt filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
    )
    print("wrote requirements.txt, README.md, Dockerfile, .gitattributes")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nbundle ready at {OUT}  ({total / 1024 / 1024:.0f} MB)")


if __name__ == "__main__":
    main()
