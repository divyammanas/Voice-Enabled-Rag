---
title: Voice RAG Observability Panel
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Voice-RAG Observability Panel

**Task 2 submission — HH Goa 2026**

A single-file, single-process, voice-enabled RAG **observability /
engineering control panel** — not a consumer chat app. One Hindi voice
query runs through:

```
audio → STT (Sarvam AI) → normalize → hybrid retrieve (dense + lexical) → evidence gate → grounded generate
```

with every intermediate artifact and every stage's latency shown, so you
can see *why* it answered, refused, or errored — not just the final text.

## Files

```
├── app.py              # everything: STT, normalize, retrieve, evidence gate, generate, Gradio UI
├── requirements.txt
└── README.md
```

Deliberately one file. There's no separate trace/dataclass abstraction —
`run_pipeline()` builds a plain local `timings_ms` dict and local
variables per call, times each stage with `time.perf_counter()`, and
renders the result. Nothing is stored at module level except the
read-only corpus/index built once at import, so back-to-back requests
never share state.

## Why single-process

The Gradio UI and the pipeline logic run in the exact same Python
process — `run_pipeline()` is called directly from the button's
`.click()` handler, no internal API hop. The latency panel reflects
real pipeline cost, not network overhead between services.

## Pipeline statuses

| Status | Meaning |
|---|---|
| `answered` | Evidence gate passed, generation was grounded — answer shown. |
| `refused_no_evidence` | Retrieval came back too weak / out-of-corpus — generation is **skipped entirely** (check `timings_ms` — `generation` won't even appear). |
| `refused_ungrounded` | Generation ran but didn't sufficiently overlap with the retrieved evidence — refused rather than shown. |
| `error` | STT failed, audio was empty/corrupt, or an unexpected exception occurred — caught cleanly, never crashes the UI. |

## Setup

```bash
pip install -r requirements.txt
export SARVAM_API_KEY="your-key-here"   # set as a Space secret on Hugging Face
python app.py
```

## Usage

1. Record or upload a short **Hindi** audio clip.
2. Click **Run Pipeline**.
3. Read the five panels: **Transcript**, **Retrieved Evidence** (passage id, fused score to 3dp, `dense`/`lexical` source, text preview), **Answer** (grounded answer or a bold `**REFUSED**` reason), **Status** (machine-readable outcome + request id), **Latency** (per-stage ms + total).

## Notes on the demo retrieval/generation logic

The corpus, "dense" retrieval (hashed bag-of-words cosine similarity), and
generation (extractive stitch + token-overlap grounding check) are all
deterministic and dependency-light, so the demo has predictable,
near-zero-latency behavior on those stages with no GPU or external LLM
key needed. Swap `_embed()` for a real encoder + FAISS index, and
`_compose_answer()` for a real LLM call, without touching anything else
in the file.

## Design constraints honored

- Gradio only — no React/Streamlit/FastAPI.
- Single process, single file — zero API hops, zero extra abstraction layers.
- No auth, no accounts, no multi-turn history/session state.
- `gr.themes.Soft()` only — no custom CSS.
- Hindi-only, no language toggle.
- Stateless per request — fresh local state every call; no shared mutable state anywhere.
