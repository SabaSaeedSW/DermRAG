"""Evaluation for all three agents, reading from results/pipeline_results.jsonl."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "pipeline_results.jsonl"
FIGURES_DIR = ROOT / "results" / "figures"
LABELS = ["benign", "malignant"]


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"no results at {RESULTS_PATH}\n"
            "run first:  python3 src/pipeline.py --no-reason"
        )
    rows = [json.loads(line) for line in open(RESULTS_PATH) if line.strip()]
    df = pd.DataFrame(rows)
    # A resumed/re-run file can contain duplicates; keep the most recent per image.
    return df.drop_duplicates(subset="image_id", keep="last").reset_index(drop=True)


# classifier

def eval_classifier(df: pd.DataFrame) -> None:
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    print(f"=== Agent 1: classifier ({len(df)} test images) ===\n")
    print(f"accuracy: {accuracy_score(y_true, y_pred):.3f}")

    p_malignant = df["class_probabilities"].apply(lambda d: d["malignant"])
    auc = roc_auc_score((y_true == "malignant").astype(int), p_malignant)
    print(f"ROC AUC:  {auc:.3f}   (threshold-independent, unlike the metrics below)\n")

    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS],
                       columns=[f"pred_{x}" for x in LABELS]))

    tn, fp, fn, tp = cm.ravel()
    print(f"\nmissed malignant cases (false negatives): {fn}")
    print(f"false alarms (false positives):           {fp}")
    print("in this application a false negative is the costlier error.")

    # Per-dx breakdown: which diagnoses does the binary classifier actually miss?
    print("\nrecall by underlying diagnosis:")
    for dx, group in df.groupby("true_dx"):
        correct = (group["true_label"] == group["predicted_label"]).sum()
        print(f"  {dx:5s} {correct:4d}/{len(group):<4d} = {correct / len(group):.1%}  "
              f"(n={len(group)}, {group['true_label'].iloc[0]})")

    _plot_confusion(cm)


def _plot_confusion(cm: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [f"pred\n{x}" for x in LABELS])
    ax.set_yticks([0, 1], [f"true\n{x}" for x in LABELS])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("Agent 1 confusion matrix")
    fig.tight_layout()
    out = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {out}")


# retrieval

def eval_retrieval(df: pd.DataFrame) -> None:
    k = len(df["retrieved"].iloc[0])
    print(f"\n=== Agent 2: retrieval precision@{k} ({len(df)} queries) ===\n")

    def dx_hits(row):
        return sum(1 for r in row["retrieved"] if r["dx"] == row["true_dx"])

    def label_hits(row):
        return sum(1 for r in row["retrieved"] if r["label"] == row["true_label"])

    df = df.copy()
    df["dx_hits"] = df.apply(dx_hits, axis=1)
    df["label_hits"] = df.apply(label_hits, axis=1)

    print(f"precision@{k} (exact 7-class dx match):   {df['dx_hits'].sum() / (len(df) * k):.1%}")
    print(f"precision@{k} (binary benign/malignant): {df['label_hits'].sum() / (len(df) * k):.1%}")
    print("\nthe binary figure is inflated by nv dominating the index : read the per-class table.\n")

    print(f"per-class precision@{k} (exact dx):")
    rows = []
    for dx, group in df.groupby("true_dx"):
        prec = group["dx_hits"].sum() / (len(group) * k)
        rows.append((dx, prec, len(group)))
        print(f"  {dx:5s} {prec:6.1%}   (n={len(group)} queries)")

    macro = float(np.mean([r[1] for r in rows]))
    print(f"\nmacro-average across classes: {macro:.1%}")
    print("macro-average is the honest headline here: it weights rare classes equally,")
    print("so nv cannot mask failures on df / bcc.")

    _plot_retrieval(rows, k)


def _plot_retrieval(rows: list[tuple], k: int) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="steelblue")
    ax.set_xlabel(f"precision@{k} (exact dx match)")
    ax.set_xlim(0, 1)
    ax.set_title("Agent 2 retrieval quality by class")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.01, i, f"{r[1]:.0%} (n={r[2]})", va="center", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "retrieval_precision.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


# agreement

def eval_agreement(df: pd.DataFrame) -> None:
    """Where classifier and retrieved evidence disagree : the cases Agent 3 must flag."""
    print(f"\n=== Agents 1 vs 2: evidence agreement ({len(df)} cases) ===\n")

    df = df.copy()
    df["correct"] = df["true_label"] == df["predicted_label"]
    # "Disagreement" = fewer than half the neighbours share the predicted label.
    df["disagrees"] = df["neighbour_agreement"] < 0.5

    n_dis = int(df["disagrees"].sum())
    print(f"cases where retrieval contradicts the classifier: {n_dis} ({n_dis / len(df):.1%})")

    if n_dis:
        acc_dis = df.loc[df["disagrees"], "correct"].mean()
        acc_agree = df.loc[~df["disagrees"], "correct"].mean()
        print(f"  classifier accuracy when evidence agrees:    {acc_agree:.1%}")
        print(f"  classifier accuracy when evidence disagrees: {acc_dis:.1%}")
        if acc_dis < acc_agree:
            print("\n  -> disagreement is a useful warning signal: the classifier is")
            print("     measurably less reliable on these cases.")
        else:
            print("\n  -> disagreement does NOT predict classifier error here;")
            print("     retrieval noise, not classifier uncertainty, may explain it.")

    worst = df[df["disagrees"] & ~df["correct"]].nlargest(5, "confidence")
    if not worst.empty:
        print("\nconfidently wrong AND contradicted by evidence (best blog cases):")
        for _, r in worst.iterrows():
            print(f"  {r['image_id']}  true={r['true_dx']:5s}({r['true_label']:9s})  "
                  f"pred={r['predicted_label']:9s} conf={r['confidence']:.2f}  "
                  f"agree={r['neighbour_agreement']:.0%}")


# grad-cam

def eval_gradcam(n: int, seed: int) -> None:
    """Check whether the classifier attends to the lesion or to rulers/ink artifacts."""
    import torch
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    import agent1_classifier as a1

    device = a1.get_device()
    model = a1.load_model(device)
    # ResNet18: last conv stage carries the spatial evidence for the decision.
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    df = load_results()
    df = df.copy()
    df["correct"] = df["true_label"] == df["predicted_label"]
    # Sample both correct and incorrect predictions : shortcuts show up in both.
    wrong = df[~df["correct"]].sample(n=min(n // 2, (~df["correct"]).sum()), random_state=seed)
    right = df[df["correct"]].sample(n=min(n - len(wrong), df["correct"].sum()), random_state=seed)
    sample = pd.concat([wrong, right])

    paths = pd.read_csv(ROOT / "data" / "splits" / "test.csv").set_index("image_id")["image_path"]
    transform = a1.get_transforms(train=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(sample), figsize=(2.2 * len(sample), 5))
    if len(sample) == 1:
        axes = axes.reshape(2, 1)

    for col, (_, row) in enumerate(sample.iterrows()):
        image = Image.open(paths[row["image_id"]]).convert("RGB").resize((224, 224))
        tensor = transform(image).unsqueeze(0).to(device)
        target = ClassifierOutputTarget(LABELS.index(row["predicted_label"]))
        grayscale = cam(input_tensor=tensor, targets=[target])[0]

        axes[0, col].imshow(image)
        axes[0, col].set_title(
            f"{row['true_dx']} -> {row['predicted_label'][:4]}\n"
            f"{'OK' if row['correct'] else 'MISS'} conf={row['confidence']:.2f}",
            fontsize=8,
        )
        axes[1, col].imshow(image)
        axes[1, col].imshow(grayscale, cmap="jet", alpha=0.5)
        for ax in (axes[0, col], axes[1, col]):
            ax.axis("off")

    fig.suptitle("Grad-CAM: is the model looking at the lesion, or at an artifact?", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / "gradcam.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")
    print("\ninspect manually: heat centred on the lesion is good; heat on a ruler,")
    print("ink marking, vignette, or image border indicates shortcut learning.")


# rationales

def eval_rationales(df: pd.DataFrame, n: int, seed: int) -> None:
    """Emit a scoring sheet of generated rationales for manual / LLM-judge review."""
    has = df[df["rationale"].notna() & (df["rationale"] != "")]
    if has.empty:
        print("\nno rationales in results yet : run the pipeline without --no-reason:")
        print("  python3 src/pipeline.py -n 200 --stratified")
        return

    print(f"\n=== Agent 3: rationale sample ({len(has)} available) ===\n")
    # Over-sample disagreement cases : they're where groundedness actually gets tested.
    dis = has[has["neighbour_agreement"] < 0.5]
    agree = has[has["neighbour_agreement"] >= 0.5]
    take_dis = dis.sample(n=min(n // 2, len(dis)), random_state=seed)
    take_agree = agree.sample(n=min(n - len(take_dis), len(agree)), random_state=seed)
    sample = pd.concat([take_dis, take_agree])

    out_path = ROOT / "results" / "rationale_scoring_sheet.md"
    lines = [
        "# Rationale scoring sheet",
        "",
        "Score each 1-5. **Groundedness**: does it reference only the classifier output and",
        "retrieved cases given, without inventing findings? **Disagreement flagging**: when",
        "evidence contradicts the classifier, does it say so explicitly?",
        "",
    ]
    for _, r in sample.iterrows():
        neighbours = ", ".join(f"{x['dx']}({x['similarity']:.2f})" for x in r["retrieved"])
        lines += [
            f"## {r['image_id']}  (true: {r['true_dx']} / {r['true_label']})",
            "",
            f"- classifier: **{r['predicted_label']}** @ {r['confidence']:.0%}",
            f"- neighbours: {neighbours}",
            f"- agreement: {r['neighbour_agreement']:.0%}"
            f"{'  <- DISAGREEMENT CASE' if r['neighbour_agreement'] < 0.5 else ''}",
            f"- model: {r.get('reasoner_model', 'n/a')}",
            "",
            "> " + str(r["rationale"]).replace("\n", "\n> "),
            "",
            "| groundedness (1-5) | disagreement flagged (y/n/na) | notes |",
            "|---|---|---|",
            "|  |  |  |",
            "",
        ]
    out_path.write_text("\n".join(lines))
    print(f"wrote {len(sample)} rationales -> {out_path}")
    print(f"({len(take_dis)} disagreement cases, {len(take_agree)} agreement cases)")


# cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all",
                        choices=["all", "classifier", "retrieval", "agreement",
                                 "gradcam", "rationales"])
    parser.add_argument("-n", type=int, default=8, help="samples for gradcam / rationales")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "gradcam":
        eval_gradcam(args.n, args.seed)
    else:
        results = load_results()
        if args.mode in ("all", "classifier"):
            eval_classifier(results)
        if args.mode in ("all", "retrieval"):
            eval_retrieval(results)
        if args.mode in ("all", "agreement"):
            eval_agreement(results)
        if args.mode == "rationales":
            eval_rationales(results, args.n, args.seed)
