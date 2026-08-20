"""Orchestrates Agent 1 (classifier) -> Agent 2 (retriever) -> Agent 3 (reasoner)."""

import os

# Must be set before torch or faiss is imported anywhere in the process, including
# transitively via the agent modules below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

import agent1_classifier
import agent2_retriever

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_PATH = ROOT / "results" / "pipeline_results.jsonl"


@dataclass
class PipelineResult:
    image_id: str
    true_dx: str
    true_label: str
    predicted_label: str
    confidence: float
    class_probabilities: dict
    retrieved: list
    neighbour_agreement: float
    rationale: str | None
    reasoner_stop_reason: str | None
    reasoner_provider: str | None = None
    reasoner_model: str | None = None


def neighbour_agreement(predicted_label: str, retrieved: list[dict]) -> float:
    """Fraction of retrieved neighbours whose benign/malignant label matches the prediction."""
    if not retrieved:
        return 0.0
    matches = sum(1 for r in retrieved if r["label"] == predicted_label)
    return matches / len(retrieved)


def load_queries(n: int | None, stratified: bool, seed: int) -> pd.DataFrame:
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    if n is None:
        return test_df
    if stratified:
        per_class = max(1, n // test_df["dx"].nunique())
        return pd.concat([
            group.sample(n=min(per_class, len(group)), random_state=seed)
            for _, group in test_df.groupby("dx")
        ], ignore_index=True)
    return test_df.sample(n=min(n, len(test_df)), random_state=seed)


def already_processed(need_rationale: bool) -> set[str]:
    """Image ids that don't need reprocessing.

    When the reasoner is enabled, a row written by an earlier --no-reason pass is
    *not* done : it still needs its rationale.
    """
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if need_rationale and not row.get("rationale"):
                continue
            done.add(row["image_id"])
    return done


def run(n: int | None, k: int, stratified: bool, seed: int, reason: bool, resume: bool,
        provider: str | None = None, delay_s: float = 0.0) -> None:
    queries = load_queries(n, stratified, seed)

    done = already_processed(need_rationale=reason) if resume else set()
    if done:
        queries = queries[~queries["image_id"].isin(done)]
        print(f"resuming: {len(done)} already done, {len(queries)} remaining")

    if queries.empty:
        print("nothing to do")
        return

    device = agent1_classifier.get_device()
    print(f"loading agent 1 (classifier) on {device}...")
    classifier = agent1_classifier.load_model(device)

    print("loading agent 2 (retriever)...")
    retriever = agent2_retriever.Retriever()

    reasoner = None
    if reason:
        import agent3_reasoner

        chosen = provider or agent3_reasoner.default_provider()
        reasoner = agent3_reasoner.get_reasoner(chosen, delay_s=delay_s)
        print(f"loading agent 3 (reasoner, {reasoner.provider}/{reasoner.model})...")
    else:
        print("agent 3 disabled (--no-reason): running agents 1+2 only, no API calls")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(queries)

    with open(RESULTS_PATH, "a") as out:
        for i, (_, row) in enumerate(queries.iterrows(), start=1):
            prediction = agent1_classifier.predict(classifier, row["image_path"], device)
            retrieved = retriever.retrieve(row["image_path"], k=k)

            rationale, stop_reason = None, None
            if reasoner is not None:
                reasoning = reasoner.reason(prediction, retrieved, image_id=row["image_id"])
                rationale = reasoning["rationale"]
                stop_reason = reasoning["stop_reason"]
                if not reasoning["ok"]:
                    # Don't persist a failed case : leaving it out means --resume retries it
                    # rather than treating the error string as a finished rationale.
                    print(f"[{i}/{n_total}] {row['image_id']} SKIPPED (reasoner failed)")
                    continue

            result = PipelineResult(
                image_id=row["image_id"],
                true_dx=row["dx"],
                true_label=row["label"],
                predicted_label=prediction["predicted_label"],
                confidence=prediction["confidence"],
                class_probabilities=prediction["class_probabilities"],
                retrieved=retrieved,
                neighbour_agreement=neighbour_agreement(prediction["predicted_label"], retrieved),
                rationale=rationale,
                reasoner_stop_reason=stop_reason,
                reasoner_provider=reasoner.provider if reasoner else None,
                reasoner_model=reasoner.model if reasoner else None,
            )
            out.write(json.dumps(asdict(result)) + "\n")
            out.flush()

            correct = "ok " if result.predicted_label == row["label"] else "MISS"
            print(
                f"[{i}/{n_total}] {row['image_id']} dx={row['dx']:5s} "
                f"pred={result.predicted_label:9s} conf={result.confidence:.2f} "
                f"agree={result.neighbour_agreement:.0%} {correct}"
            )

    print(f"\nwrote results -> {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=None, help="number of test images (default: all)")
    parser.add_argument("--k", type=int, default=5, help="neighbours to retrieve")
    parser.add_argument("--stratified", action="store_true", help="sample evenly across dx classes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-reason", dest="reason", action="store_false",
                        help="skip Agent 3 (no API cost)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="reprocess images already in the results file")
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default=None,
                        help="reasoner provider (default: whichever key is set)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds to sleep between reasoner calls (free-tier rate limits)")
    args = parser.parse_args()

    run(n=args.n, k=args.k, stratified=args.stratified, seed=args.seed,
        reason=args.reason, resume=args.resume, provider=args.provider, delay_s=args.delay)
