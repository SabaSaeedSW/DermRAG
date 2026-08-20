"""Streamlit demo for DermRAG: upload a lesion image, see all three agents work."""

import os

# Must precede any torch / faiss import (they link conflicting OpenMP runtimes on macOS).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import agent1_classifier
import agent2_retriever
import agent3_reasoner

st.set_page_config(page_title="DermRAG", page_icon="🔬", layout="wide")

DISCLAIMER = (
    "**Research and educational demo : not a diagnostic tool.** "
    "This model was trained on a single public dataset, makes frequent mistakes, and "
    "must not be used to make decisions about anyone's health. If you have a concern "
    "about a skin lesion, see a qualified clinician."
)


@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier():
    device = agent1_classifier.get_device()
    return agent1_classifier.load_model(device), device


@st.cache_resource(show_spinner="Loading retrieval index…")
def load_retriever():
    return agent2_retriever.Retriever()


def resolve(path: str) -> Path:
    """Image paths are absolute in the dev repo and relative in the deployed bundle."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@st.cache_data(show_spinner=False)
def load_examples() -> dict:
    """A few held-out test images, one per diagnosis, so visitors can try the demo."""
    split = ROOT / "data" / "splits" / "test.csv"
    if not split.exists():
        return {}
    df = pd.read_csv(split)
    picks = {}
    for dx, group in df.groupby("dx"):
        row = group.iloc[0]
        if resolve(row["image_path"]).exists():
            picks[row["image_id"]] = {
                "path": str(resolve(row["image_path"])), "dx": dx, "label": row["label"],
            }
    return picks


def has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def render_probabilities(probs: dict) -> None:
    df = pd.DataFrame({"probability": probs}).sort_values("probability", ascending=False)
    st.bar_chart(df, height=180)


def main() -> None:
    st.title("🔬 DermRAG")
    st.caption("A three-agent pipeline for dermatoscopic lesion triage : "
               "classifier, image retrieval, and grounded reasoning.")
    st.error(DISCLAIMER)

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. **Classifier** : a fine-tuned ResNet18 predicts benign vs malignant.\n"
            "2. **Retriever** : CLIP embeddings + FAISS find the most visually similar "
            "*labelled* cases from the training set.\n"
            "3. **Reasoner** : an LLM writes a rationale grounded strictly in the outputs "
            "of steps 1 and 2."
        )
        st.divider()
        st.subheader("Known limitations")
        st.markdown(
            "- Retrieval is far more reliable for common moles (`nv`) than for rare "
            "classes; macro-averaged precision@5 is only ~35%.\n"
            "- The classifier is tuned to favour recall, so **false alarms are common**.\n"
            "- Retrieved neighbours are nearest points in a generic image-embedding "
            "space : they are evidence to weigh, not votes to count."
        )
        st.divider()
        k = st.slider("Neighbours to retrieve", 3, 10, 5)
        use_llm = st.checkbox("Generate LLM rationale", value=has_api_key(),
                              disabled=not has_api_key())
        if not has_api_key():
            st.info("Set `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` to enable the reasoner.")

    uploaded = st.file_uploader("Upload a dermatoscopic image", type=["jpg", "jpeg", "png"])

    examples = load_examples()
    source_name = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        source_name = uploaded.name
    elif examples:
        st.caption("…or try one of these held-out test images:")
        choice = st.selectbox(
            "Example image", [":"] + list(examples),
            format_func=lambda c: c if c == ":" else f"{c}  ({examples[c]['dx']})",
            label_visibility="collapsed",
        )
        if choice == ":":
            st.info("Upload an image or pick an example to run the pipeline.")
            return
        image = Image.open(examples[choice]["path"]).convert("RGB")
        source_name = choice
        st.caption(f"Ground truth for this example: **{examples[choice]['dx']}** "
                   f"({examples[choice]['label']}) : shown only because it is a labelled "
                   "test case; the pipeline never sees it.")
    else:
        st.info("Upload a dermatoscopic image to run the pipeline.")
        return

    # Write the query to the system temp dir: the app directory is not reliably
    # writable on hosted platforms.
    tmp_path = Path(tempfile.gettempdir()) / "dermrag_query.jpg"
    image.save(tmp_path)

    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Query image", use_container_width=True)

    with right:
        st.subheader("Agent 1 : Classifier")
        model, device = load_classifier()
        prediction = agent1_classifier.predict(model, str(tmp_path), device)

        label = prediction["predicted_label"]
        p_mal = prediction["class_probabilities"]["malignant"]
        thr = prediction.get("threshold", 0.5)
        detail = f"probability of malignant {p_mal:.1%} · threshold {thr:.0%}"
        if label == "malignant":
            st.warning(f"Prediction: **{label}** — {detail}")
        else:
            st.success(f"Prediction: **{label}** — {detail}")
        st.caption(
            f"The threshold sits below 50% on purpose: this is a triage tool, so it is "
            f"tuned to catch ~90% of malignancies at the cost of more false alarms."
        )
        render_probabilities(prediction["class_probabilities"])

    st.divider()
    st.subheader(f"Agent 2 : {k} most similar labelled cases")
    retriever = load_retriever()
    with st.spinner("Searching the reference index…"):
        retrieved = retriever.retrieve(str(tmp_path), k=k)

    agreement = sum(1 for r in retrieved if r["label"] == label) / len(retrieved)
    cols = st.columns(len(retrieved))
    for col, case in zip(cols, retrieved):
        with col:
            st.image(Image.open(resolve(case["image_path"])).convert("RGB"),
                     use_container_width=True)
            full = agent3_reasoner.DX_FULL_NAME.get(case["dx"], case["dx"])
            st.caption(f"**{case['dx']}** : {full.split('(')[0].strip()}\n\n"
                       f"similarity {case['similarity']:.2f}")

    if agreement < 0.5:
        st.warning(f"⚠️ **Evidence disagrees with the classifier** : only "
                   f"{agreement:.0%} of retrieved cases share the predicted label. "
                   "On the held-out test set, classifier accuracy fell from 92% to 46% "
                   "on cases like this, so treat this prediction with extra caution.")
    else:
        st.info(f"{agreement:.0%} of retrieved cases share the predicted label. "
                "Note that agreement is **not** confirmation: the classifier and the "
                "retriever read the same image features, so they can be wrong together : "
                "a missed melanoma can look like a mole to both.")

    st.divider()
    st.subheader("Agent 3 : Rationale")
    if not use_llm:
        st.caption("LLM rationale disabled. Enable it in the sidebar with an API key set.")
    else:
        with st.spinner("Generating rationale…"):
            reasoner = agent3_reasoner.get_reasoner(agent3_reasoner.default_provider())
            result = reasoner.reason(prediction, retrieved, image_id=source_name)
        if result["ok"]:
            st.markdown(result["rationale"])
            st.caption(f"{result['provider']}/{result['model']} · {result['latency_s']}s")
        else:
            st.error("The reasoner call failed (often a free-tier rate limit). "
                     "Wait a moment and try again.")
            st.caption(result["rationale"])

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
