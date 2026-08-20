# DermRAG

> **Research and educational demo, not a diagnostic tool.** This system was trained on a
> single public dataset, makes frequent mistakes, and must not be used to make decisions
> about anyone's health. Nothing in this repository is medical advice.

A small, single-developer multi-agent pipeline for dermatoscopic lesion triage, inspired by
the UW RAIVN Lab's *PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic
Decision-Making* (2025). Where PathFinder applies the idea to histopathology, this applies a
scaled-down version to dermatoscopic images from HAM10000.

The project sets out to demonstrate three things:

1. a working **multi-agent pipeline** rather than one model doing everything end to end;
2. **image-based RAG** — retrieval over a vector index of image embeddings, not text;
3. an **LLM that reasons over retrieved evidence** instead of hallucinating a diagnosis.

The most interesting result is about the third point, and it is not entirely flattering.
See [Findings](#findings).

---

## Architecture

```
                  ┌──────────────────────────────┐
  lesion image ──▶│ Agent 1  Classifier          │──▶ p(benign), p(malignant)
        │         │ ResNet18, fine-tuned         │
        │         └──────────────────────────────┘
        │         ┌──────────────────────────────┐
        └────────▶│ Agent 2  Retriever           │──▶ k most similar *labelled* cases
                  │ CLIP ViT-B-32 + FAISS        │    with diagnoses + similarity
                  └──────────────────────────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │ Agent 3  Reasoner            │──▶ plain-language rationale,
                  │ Gemini or Claude             │    grounded strictly in the above
                  └──────────────────────────────┘
```

Agent 1 gives a prediction but no justification. Agent 2 supplies **inspectable evidence**:
real, diagnosed cases that look like the query. Agent 3 reasons over both and is explicitly
instructed never to diagnose beyond that evidence. The separation is the point: a single
model asked to classify *and* explain will happily invent a justification for whatever it
predicted.

---

## Pipeline, start to end

### Phase 0 — Data preparation (`src/data_prep.py`)

HAM10000: **10,015 dermatoscopic images** across 7 diagnoses, distributed as two image
folders plus a metadata CSV. The script combines the folders (via symlink, no duplication),
derives a binary label, and writes the splits.

| Code | Diagnosis | Class |
|---|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma | malignant |
| `bcc` | Basal cell carcinoma | malignant |
| `mel` | Melanoma | malignant |
| `bkl` | Benign keratosis-like lesions | benign |
| `df` | Dermatofibroma | benign |
| `nv` | Melanocytic nevi (moles) | benign |
| `vasc` | Vascular lesions | benign |

**The single most important correctness step is splitting by `lesion_id`, not `image_id`.**
Some lesions are photographed more than once. Splitting naively puts two images of the *same
lesion* in train and test, so the model is graded on something it has effectively memorised
and every downstream number is inflated. The split groups by lesion first, and the code
asserts no lesion appears in two splits.

| Split | Images | Lesions | Malignant |
|---|---|---|---|
| train | 7,002 | 5,229 | 19.4% |
| val | 1,519 | 1,120 | 18.3% |
| test | 1,494 | 1,121 | 21.3% |

`nv` alone is roughly two-thirds of the dataset, so class imbalance drives most of the design
decisions that follow.

### Phase 1 — Agent 1, the classifier (`src/agent1_classifier.py`)

ImageNet-pretrained **ResNet18** with a fresh binary head, fine-tuned at 224×224. Flips,
rotation, and colour jitter for augmentation (dermatoscopic images have no canonical "up").
Class imbalance handled by `WeightedRandomSampler` or class-weighted loss. Inference returns
the **full probability distribution**, not just the top label, because Agent 3 needs the
confidence spread to reason about uncertainty.

### Phase 2 — Agent 2, the retriever (`src/agent2_retriever.py`)

Every train+val image (**8,521**) is embedded with **CLIP ViT-B-32** (`laion2b_s34b_b79k`,
not fine-tuned) into a 512-dimensional vector, indexed in a FAISS `IndexFlatIP` over
L2-normalised vectors, which makes inner product equivalent to cosine similarity. A query is
embedded the same way and the k nearest neighbours are returned with their real diagnoses.

**Test images are deliberately excluded from the index.** Include them and every query
retrieves itself as its own nearest neighbour, and the numbers become meaningless.

### Phase 3 — Agent 3, the reasoner (`src/agent3_reasoner.py`)

Takes Agent 1's distribution and Agent 2's neighbours and writes a 3–4 sentence rationale.
The prompt requires it to state the prediction and confidence, say whether the retrieved
cases support or contradict it, **explicitly flag disagreement**, and add nothing not
grounded in the evidence supplied. Every response ends with a research-demo disclaimer.

Two providers sit behind one interface (`--provider gemini|anthropic`) so rationale quality,
latency, and cost can be compared across model families. Only text is sent to the API —
probabilities and diagnosis labels, never images.

One line was added to the plan's original prompt after seeing the retrieval numbers:

> The retrieved cases are nearest neighbours in a generic image-embedding space, not a
> diagnostic vote. Treat them as context to weigh, not as a count to tally.

### Phase 4 — Integration (`src/pipeline.py`)

Chains all three agents over the test set, one JSON line per case, with sampling,
stratification, resume, and a `--no-reason` mode that runs agents 1–2 with no API cost. It
also records `neighbour_agreement`: the fraction of retrieved neighbours sharing the
predicted label. That field turned out to matter more than expected.

### Phase 5 — Evaluation (`src/evaluate.py`)

Classifier metrics, retrieval precision@k, agreement analysis, Grad-CAM shortcut checks, and
a rationale scoring sheet for manual or LLM-judge review.

### Phase 6 — Demo (`app.py`)

Streamlit app showing all three agents on one page, with held-out example images so it is
usable without a dermatoscopic photo, and the disclaimer at top and bottom.

---

## Results

All figures are on the **held-out test split (1,494 images, 318 malignant)**.

### Agent 1

| Metric | Value |
|---|---|
| Accuracy | 0.809 |
| ROC AUC | **0.915** |
| Malignant recall | 0.899 |
| Malignant precision | 0.530 |
| Missed malignancies | **32 / 318** |

These are after promoting the tuned checkpoint and applying a decision threshold of
0.33 chosen on validation under a 90% recall floor. The untuned baseline scored 0.756
accuracy / 0.905 AUC and missed 36.

Recall by underlying diagnosis exposes where it really fails:

```
akiec 96.8%   mel  88.8%   bcc  86.8%      malignant classes: strong
nv    84.7%   df   57.1%   vasc 52.4%      benign classes: weaker
bkl   41.4%  (n=152)
```

The model is tuned to favour recall, so false alarms are common by design: a missed melanoma
is a worse error than an unnecessary second look. `bkl` at 41.4% accounts for most of the
254 false alarms.

### Agent 2

| Metric | Value |
|---|---|
| precision@5, exact 7-class dx | 68.6% |
| precision@5, binary | 79.4% |
| **precision@5, macro-averaged** | **35.2%** |

```
nv    87.1%    bkl  35.4%    mel  33.2%    bcc 29.4%
akiec 26.0%    vasc 23.8%    df   11.4%
```

The gap between 68.6% and 35.2% *is* the finding. An unstratified metric here mostly measures
how much of the index is moles. Generic CLIP embeddings, never trained on dermatoscopy, find
"a roundish pigmented blob" similar to any other roundish pigmented blob, and the index is
saturated with them, so retrieval collapses toward `nv` for every rare class.

### Grad-CAM

HAM10000 is known to contain rulers and ink markings, a classic shortcut-learning trap. On
the sampled images, attention is lesion-centred, and on one `akiec` image with **purple ink
at both edges the heat sits squarely on the lesion and ignores the ink**. Suggestive but not
conclusive: this was 8 images, and two correct-but-suspicious panels put heat on plain skin.

---

## Findings

### 1. Disagreement is a real reliability signal

Splitting the test set by whether retrieval agrees with the classifier:

| | Classifier accuracy |
|---|---|
| Retrieval **agrees** (1,131 cases) | **92.2%** |
| Retrieval **disagrees** (363 cases, 24%) | **45.5%** |

When the second opinion contradicts the first, the classifier is roughly **2.5× more likely
to be wrong**. This is the central result: the multi-agent structure surfaces information a
single model does not expose, and it justifies the disagreement-flagging requirement in
Agent 3's prompt. It is also computable without an LLM, so it works as a plain metric.

A real example the pipeline caught — a benign vascular lesion the classifier called malignant
at 96% confidence, with zero of five neighbours malignant:

> "The classifier predicted the lesion is malignant with a high confidence of 95.5%. However,
> the top visually similar reference cases retrieved from the dataset are all benign...
> Because the retrieved reference cases strongly contradict the classifier's high-malignancy
> prediction, this discrepancy flags the case as warranting closer review."

### 2. Agreement is *not* a safety signal — the important caveat

The converse does not hold, and this is the limitation worth taking seriously:

```
missed melanomas:                                 21
  where retrieval AGREED with the wrong call:     16   (76%)
high-agreement cases that are nonetheless wrong:  88   (7.8%)
```

**76% of missed melanomas land in the blind spot where both agents fail together.** The two
agents are not independent: they read the same image through similar visual features, so when
a melanoma looks like a mole, both are fooled the same way and their agreement amplifies the
error rather than catching it. In one demo run the system missed a melanoma and told the user
there were "no conflicting signals... to warrant elevated concern."

This is an architectural limitation, not a prompt bug. For the ensemble to add *safety*
rather than just confidence, the retriever has to fail **differently** from the classifier —
which argues for a dermatology-tuned encoder or a class-balanced index, not better prompting.

### 3. Tuning helped less than fixing a metric bug

The original training loop saved the checkpoint with the best malignant recall. That metric is
gameable: a model predicting "malignant" for everything scores 100%. It duly saved epoch 1.
Selecting on ROC AUC instead, and training longer, moved test AUC 0.905 → 0.915. Hyperparameter
search contributed almost nothing by comparison: across architecture, learning rate, weight
decay, and imbalance strategy, validation AUC moved by **0.002**, and the same configuration
scored 0.9268 and 0.9225 on two separate runs — run-to-run variance exceeded every effect being
measured. EfficientNet-B0 was consistently slightly worse than ResNet18.

Choosing the operating point mattered far more than choosing the model. Measured as **precision
at matched recall**, the promoted model gains **+8 points of precision at 90% recall**. But at
the naive 0.5 threshold it misses **49** cancers; at the tuned 0.33 threshold, **32**. Selecting
on F1 — which looks best on paper — would have missed **74**. F1 is the wrong objective for an
imbalanced, safety-critical problem.

---

## Limitations

- Binary benign/malignant only; the 7-class extension is not built.
- Retrieval is near-useless for rare classes (`df` 11%, `vasc` 24%).
- The reasoner is never independently verified; it can only be as good as its inputs, and
  Finding 2 shows those inputs can be confidently wrong together.
- Grad-CAM evidence is from 8 images.
- Single dataset, single centre, no external validation, and HAM10000 is not representative
  of all skin types.

---

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place HAM10000 (`HAM10000_images_part_1/`, `part_2/`, `HAM10000_metadata.csv`) under
`data/raw/`, then:

```bash
python3 src/data_prep.py                      # build lesion-level splits
python3 src/agent1_classifier.py --epochs 10  # train the classifier
python3 src/agent2_retriever.py --action build  # build the CLIP/FAISS index
python3 src/pipeline.py --no-reason           # run agents 1+2 over the test set
python3 src/evaluate.py all                   # metrics and figures
```

For Agent 3, set `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`, then:

```bash
python3 src/pipeline.py -n 200 --stratified --delay 2
python3 src/evaluate.py rationales -n 20
streamlit run app.py
```

Note that `gemini-3.7-flash` allows only 20 free requests per day; the default is
`gemini-3.5-flash-lite`, which has a far larger free quota.

---

## Repository layout

```
src/data_prep.py          combine folders, lesion-level split
src/agent1_classifier.py  train / load / predict
src/agent2_retriever.py   CLIP embeddings, FAISS index, retrieval
src/agent3_reasoner.py    provider-agnostic reasoner (Gemini, Claude)
src/pipeline.py           orchestrates all three agents
src/evaluate.py           metrics, Grad-CAM, rationale scoring sheet
src/tune.py               threshold and hyperparameter tuning
app.py                    Streamlit demo
```

Data, models, index, and results are gitignored; they are regenerable from the commands above.

## Implementation notes

- **`faiss` and `torch` segfault together on macOS** from conflicting OpenMP runtimes. Every
  entry point sets `KMP_DUPLICATE_LIB_OK` and `OMP_NUM_THREADS` *before* either import.
- **MPS crashes loading the CLIP checkpoint**, so index building runs on CPU. The classifier
  uses MPS normally.
- Tuning selects on validation only. The test split is touched once, for the final comparison.
