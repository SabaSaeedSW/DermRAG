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
