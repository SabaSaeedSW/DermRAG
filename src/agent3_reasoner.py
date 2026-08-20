"""Agent 3: LLM reasoner that writes a grounded rationale from Agent 1 + Agent 2 output."""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "results" / "reasoner_log.jsonl"

ANTHROPIC_MODEL = "claude-opus-5"
# gemini-3.7-flash allows only 20 free requests/day; the lite model has a much
# larger free quota, which matters for a few-hundred-case evaluation run.
GEMINI_MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 2000

DX_FULL_NAME = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma (malignant, pre-cancerous)",
    "bcc": "Basal cell carcinoma (malignant)",
    "mel": "Melanoma (malignant)",
    "bkl": "Benign keratosis-like lesion (benign)",
    "df": "Dermatofibroma (benign)",
    "nv": "Melanocytic nevus / mole (benign)",
    "vasc": "Vascular lesion (benign)",
}

PROMPT_TEMPLATE = """You are assisting with an educational research demo, not a clinical diagnostic tool.
You will be given: (1) a classifier's prediction and confidence for a skin lesion image,
and (2) the diagnoses of the most visually similar reference cases from a labeled dataset.

Classifier prediction: {predicted_label}
Probability of malignant: {p_malignant:.1%}
Decision threshold: {threshold:.0%} (a lesion is called malignant at or above this value)
Full class probabilities: {class_probabilities}

Note: the threshold is deliberately set below 50% because this is a triage tool and a
missed malignancy is a worse error than a false alarm. Lowering it raises sensitivity
(more malignancies caught) at the cost of more false alarms. A malignant call at, say,
35% probability is therefore expected behaviour, not a contradiction. Report the
prediction as given, do not re-derive it from the probabilities, and do not describe
the threshold as "low sensitivity".

Top {k} visually similar reference cases:
{retrieved_block}

Write a short (3-4 sentence) plain-language rationale that:
1. States what the classifier predicted and how confident it was
2. Explains whether the retrieved reference cases support or contradict that prediction
3. If the retrieved cases disagree with the classifier, say so explicitly and flag this as
   a case warranting closer review
4. Do NOT state a diagnosis of your own beyond what is grounded in the classifier output
   and retrieved cases above. Do not speculate beyond the provided evidence.

The retrieved cases are nearest neighbours in a generic image-embedding space, not a
diagnostic vote. Treat them as context to weigh, not as a count to tally.

End your response with: "This is a research demo output, not a medical diagnosis."
"""


def format_retrieved(retrieved: list[dict]) -> str:
    lines = []
    for case in retrieved:
        full_name = DX_FULL_NAME.get(case["dx"], case["dx"])
        lines.append(f"- {full_name} (similarity: {case['similarity']:.2f})")
    return "\n".join(lines)


def build_prompt(prediction: dict, retrieved: list[dict]) -> str:
    probs = {label: f"{p:.1%}" for label, p in prediction["class_probabilities"].items()}
    return PROMPT_TEMPLATE.format(
        predicted_label=prediction["predicted_label"],
        p_malignant=prediction["class_probabilities"]["malignant"],
        threshold=prediction.get("threshold", 0.5),
        class_probabilities=probs,
        k=len(retrieved),
        retrieved_block=format_retrieved(retrieved),
    )


def log_interaction(image_id: str, provider: str, model: str, prompt: str,
                    response_text: str, stop_reason: str, latency_s: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "image_id": image_id,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "response": response_text,
        "stop_reason": stop_reason,
        "latency_s": round(latency_s, 2),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


class BaseReasoner:
    """Shared prompt-building, logging, and retry logic."""

    provider = "base"
    model = "unset"

    def __init__(self, delay_s: float = 0.0, max_retries: int = 5):
        self.delay_s = delay_s
        self.max_retries = max_retries

    def _generate(self, prompt: str) -> tuple[str, str]:
        """Return (text, stop_reason). Implemented per provider."""
        raise NotImplementedError

    @staticmethod
    def _retry_after(error: Exception) -> float | None:
        """Providers tell us how long to wait on a 429 : honour it instead of guessing."""
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
        return float(match.group(1)) if match else None

    def reason(self, prediction: dict, retrieved: list[dict], image_id: str = "unknown") -> dict:
        prompt = build_prompt(prediction, retrieved)
        start = time.monotonic()
        ok = True

        # Free tiers are rate limited; back off and retry rather than dropping a case.
        for attempt in range(self.max_retries):
            try:
                text, stop_reason = self._generate(prompt)
                ok = True
                break
            except Exception as e:  # noqa: BLE001 - provider SDKs raise different types
                if attempt == self.max_retries - 1:
                    text, stop_reason, ok = f"[error: {e}]", "error", False
                    break
                # Prefer the provider's own retry hint; fall back to exponential backoff.
                wait = self._retry_after(e) or min(2 ** attempt, 60)
                wait += 1  # small buffer so we land outside the quota window
                print(f"  rate limited, waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{self.max_retries - 1})")
                time.sleep(wait)

        latency = time.monotonic() - start
        log_interaction(image_id, self.provider, self.model, prompt, text, stop_reason, latency)

        if ok and self.delay_s:
            time.sleep(self.delay_s)

        return {
            "rationale": text,
            "stop_reason": stop_reason,
            "ok": ok,
            "provider": self.provider,
            "model": self.model,
            "latency_s": round(latency, 2),
        }


class AnthropicReasoner(BaseReasoner):
    provider = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL, **kwargs):
        super().__init__(**kwargs)
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def _generate(self, prompt: str) -> tuple[str, str]:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return "[model declined to respond for this case]", "refusal"
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text, response.stop_reason


class GeminiReasoner(BaseReasoner):
    provider = "gemini"

    def __init__(self, model: str = GEMINI_MODEL, **kwargs):
        super().__init__(**kwargs)
        from google import genai

        self.model = model
        # Reads GEMINI_API_KEY from the environment.
        self.client = genai.Client()

    def _generate(self, prompt: str) -> tuple[str, str]:
        interaction = self.client.interactions.create(model=self.model, input=prompt)
        text = interaction.output_text or ""
        status = getattr(interaction, "status", "unknown")
        if not text:
            return "[model returned no text for this case]", str(status)
        return text, str(status)


PROVIDERS = {"anthropic": AnthropicReasoner, "gemini": GeminiReasoner}


def get_reasoner(provider: str = "gemini", **kwargs) -> BaseReasoner:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[provider](**kwargs)


def default_provider() -> str:
    """Pick whichever provider has credentials available."""
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "gemini"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=default_provider())
    args = parser.parse_args()

    # Smoke test with a disagreement case: classifier says malignant, neighbours mostly benign.
    demo_prediction = {
        "predicted_label": "malignant",
        "confidence": 0.77,
        "class_probabilities": {"benign": 0.23, "malignant": 0.77},
    }
    demo_retrieved = [
        {"dx": "nv", "similarity": 0.91},
        {"dx": "nv", "similarity": 0.89},
        {"dx": "mel", "similarity": 0.88},
        {"dx": "nv", "similarity": 0.86},
        {"dx": "bkl", "similarity": 0.85},
    ]
    reasoner = get_reasoner(args.provider)
    result = reasoner.reason(demo_prediction, demo_retrieved, image_id="DEMO")
    print(result["rationale"])
    print(f"\n[{result['provider']}/{result['model']}, {result['latency_s']}s, "
          f"stop_reason={result['stop_reason']}]")
