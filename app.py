"""
app.py

Voice-RAG Observability Panel — single-file, single-process build.

Everything lives here on purpose: Gradio UI + STT + normalize + hybrid
retrieve + evidence gate + grounded generate, all running in one Python
process (no API hop between UI and pipeline). No separate trace/dataclass
abstraction — per-stage latency is just a plain dict built fresh inside
run_pipeline() on every call, so there's zero shared state and zero
cross-request leakage between back-to-back queries.

Pipeline: audio -> STT (Sarvam AI) -> normalize -> hybrid retrieve
(dense + lexical fusion) -> evidence gate -> grounded generate

Statuses: answered | refused_no_evidence | refused_ungrounded | error
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import traceback
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import requests

# ============================================================================
# 1. STT — Sarvam AI (Hindi)
# ============================================================================

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saarika:v2.5"
SARVAM_LANGUAGE_CODE = "hi-IN"
REQUEST_TIMEOUT_S = 20
MIN_VALID_AUDIO_BYTES = 512  # smaller than this is almost certainly silence/corruption


class STTError(RuntimeError):
    """Raised for any failure in the STT stage (bad audio, network, auth, empty result)."""


def transcribe(audio_path: Optional[str]) -> str:
    """Transcribe a Hindi audio clip via Sarvam AI STT. Raises STTError on any failure."""
    if not audio_path:
        raise STTError("No audio provided.")

    path = Path(audio_path)
    if not path.exists():
        raise STTError(f"Audio file not found at '{audio_path}'.")
    if path.stat().st_size < MIN_VALID_AUDIO_BYTES:
        raise STTError("Audio file is empty or too short to contain speech.")

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise STTError(
            "SARVAM_API_KEY is not set in the environment. "
            "Add it as a Hugging Face Space secret to enable transcription."
        )

    try:
        with open(path, "rb") as audio_file:
            files = {"file": (path.name, audio_file, "audio/wav")}
            data = {"model": SARVAM_MODEL, "language_code": SARVAM_LANGUAGE_CODE}
            headers = {"api-subscription-key": api_key}
            response = requests.post(
                SARVAM_STT_URL, headers=headers, files=files, data=data,
                timeout=REQUEST_TIMEOUT_S,
            )
    except requests.RequestException as exc:
        raise STTError(f"Network error while calling Sarvam STT: {exc}") from exc

    if response.status_code != 200:
        raise STTError(f"Sarvam STT API returned HTTP {response.status_code}: {response.text[:300]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise STTError("Sarvam STT API returned a non-JSON response.") from exc

    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        raise STTError("Transcription succeeded but returned an empty transcript (likely silent audio).")

    return transcript


# ============================================================================
# 2. Normalize — Hindi transcript cleanup
# ============================================================================

_FILLERS = {"उम", "उम्म", "हम्म", "अरे", "मतलब", "यानी"}
_MULTI_SPACE_RE = re.compile(r"\s+")
_PUNCT_NOISE_RE = re.compile(r"[^\w\s\u0900-\u097F?]", re.UNICODE)


class NormalizationError(ValueError):
    """Raised when the transcript is unusable after normalization (e.g. goes empty)."""


def normalize(transcript: Optional[str]) -> str:
    if transcript is None:
        raise NormalizationError("Transcript is None.")

    text = unicodedata.normalize("NFC", transcript).strip()
    text = _PUNCT_NOISE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = [tok for tok in text.split(" ") if tok and tok not in _FILLERS]
    normalized_query = " ".join(tokens)

    if not normalized_query:
        raise NormalizationError("Transcript normalized to an empty query.")

    return normalized_query


# ============================================================================
# 3. Retrieve — hybrid dense + lexical fusion over a fixed Hindi corpus
#    (index built once at import time, not per request)
# ============================================================================

_EMBED_DIM = 256
_DENSE_WEIGHT = 0.6
_LEXICAL_WEIGHT = 0.4
_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)

_CORPUS: List[Dict[str, str]] = [
    {"passage_id": "P001", "text": "भारत सरकार ने प्रधानमंत्री विद्यालक्ष्मी पोर्टल के माध्यम से शिक्षा ऋण आवेदन की प्रक्रिया को डिजिटल बना दिया है।"},
    {"passage_id": "P002", "text": "शिक्षा ऋण की अधिकतम सीमा और ब्याज दर बैंक की नीति और छात्र की पात्रता पर निर्भर करती है।"},
    {"passage_id": "P003", "text": "RAG प्रणाली किसी प्रश्न का उत्तर देने से पहले प्रासंगिक दस्तावेज़ों को पुनः प्राप्त करती है और उसी सन्दर्भ पर आधारित उत्तर उत्पन्न करती है।"},
    {"passage_id": "P004", "text": "हाइब्रिड रिट्रीवल में डेंस एम्बेडिंग खोज और लेक्सिकल कीवर्ड खोज दोनों को मिलाकर परिणामों को रीरैंक किया जाता है।"},
    {"passage_id": "P005", "text": "बैंकिंग लोकपाल (RBI Banking Ombudsman) के पास शिकायत तभी दर्ज की जा सकती है जब बैंक की शिकायत निवारण पोर्टल पर पहले समाधान न हुआ हो।"},
    {"passage_id": "P006", "text": "स्मार्ट इंडिया हैकथॉन एक राष्ट्रीय स्तर की पहल है जो छात्रों को वास्तविक समस्याओं पर तकनीकी समाधान बनाने के लिए प्रेरित करती है।"},
]


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _embed(text: str) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (stand-in for a real encoder)."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    for tok in _tokenize(text):
        digest = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = digest % _EMBED_DIM
        sign = 1.0 if (digest // _EMBED_DIM) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _build_index():
    ids, texts, token_sets, vectors = [], [], [], []
    for doc in _CORPUS:
        ids.append(doc["passage_id"])
        texts.append(doc["text"])
        token_sets.append(set(_tokenize(doc["text"])))
        vectors.append(_embed(doc["text"]))
    matrix = np.vstack(vectors) if vectors else np.zeros((0, _EMBED_DIM), dtype=np.float32)
    return ids, texts, token_sets, matrix


# Built once at import time — swap _embed()/this matrix for a real encoder + FAISS
# index in production; hybrid_retrieve()'s signature stays the same.
_PASSAGE_IDS, _PASSAGE_TEXTS, _PASSAGE_TOKENS, _DENSE_MATRIX = _build_index()


def _lexical_scores(query_tokens: set) -> np.ndarray:
    scores = np.zeros(len(_PASSAGE_TOKENS), dtype=np.float32)
    if not query_tokens:
        return scores
    for i, doc_tokens in enumerate(_PASSAGE_TOKENS):
        if not doc_tokens:
            continue
        union = query_tokens | doc_tokens
        if union:
            scores[i] = len(query_tokens & doc_tokens) / len(union)
    return scores


def hybrid_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Returns up to top_k passages: {"passage_id", "score", "text", "source": "dense"|"lexical"}."""
    if not query or not query.strip() or _DENSE_MATRIX.shape[0] == 0:
        return []

    query_vec = _embed(query)
    dense_scores = _DENSE_MATRIX @ query_vec

    query_tokens = set(_tokenize(query))
    lexical_scores = _lexical_scores(query_tokens)

    fused = _DENSE_WEIGHT * dense_scores + _LEXICAL_WEIGHT * lexical_scores
    order = np.argsort(-fused)[:top_k]

    results = []
    for i in order:
        source = "dense" if dense_scores[i] >= lexical_scores[i] else "lexical"
        results.append({
            "passage_id": _PASSAGE_IDS[i],
            "score": float(fused[i]),
            "text": _PASSAGE_TEXTS[i],
            "source": source,
        })
    return results


# ============================================================================
# 4. Evidence gate — refuse cheaply before paying for generation
# ============================================================================

EVIDENCE_SCORE_THRESHOLD = 0.12
EVIDENCE_MIN_PASSING_PASSAGES = 1


def evidence_gate_check(query: str, passages: List[Dict[str, Any]]) -> bool:
    if not query or not passages:
        return False
    passing = [p for p in passages if p.get("score", 0.0) >= EVIDENCE_SCORE_THRESHOLD]
    return len(passing) >= EVIDENCE_MIN_PASSING_PASSAGES


# ============================================================================
# 5. Generate — extractive grounded answer + hallucination check
# ============================================================================

_GROUNDING_MIN_OVERLAP = 0.35  # fraction of answer tokens that must appear in evidence


def _compose_answer(passages: List[Dict[str, Any]]) -> str:
    top = passages[:2]
    body = " ".join(p["text"] for p in top)
    citation = ", ".join(p["passage_id"] for p in top)
    return f"{body} (स्रोत: {citation})"


def _is_grounded(answer_text: str, passages: List[Dict[str, Any]]) -> bool:
    answer_tokens = set(_tokenize(answer_text))
    if not answer_tokens:
        return False
    evidence_tokens: set = set()
    for p in passages:
        evidence_tokens |= set(_tokenize(p["text"]))
    if not evidence_tokens:
        return False
    return len(answer_tokens & evidence_tokens) / len(answer_tokens) >= _GROUNDING_MIN_OVERLAP


def generate_answer(query: str, passages: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """Returns (answer_text, is_grounded). Caller decides what to do if ungrounded."""
    if not query or not passages:
        return "", False
    text = _compose_answer(passages)
    return text, _is_grounded(text, passages)


# ============================================================================
# 6. Orchestration — one function, one request, plain dict for timings
# ============================================================================

TOP_K = 5
REFUSAL_NO_EVIDENCE = "**REFUSED** - no sufficiently relevant evidence found"
REFUSAL_UNGROUNDED = "**REFUSED** - generated answer failed grounding validation"

_EMPTY_TRANSCRIPT = "_(no transcript)_"
_EMPTY_EVIDENCE = "_(no passages retrieved)_"
_EMPTY_ANSWER = "_(no answer)_"
_EMPTY_STATUS = "`pending`"
_EMPTY_LATENCY = "_(no timings recorded)_"


def run_pipeline(audio_path: Optional[str]) -> Tuple[str, str, str, str, str]:
    """
    Runs the full pipeline for one request and returns the 5 rendered panels.

    Every call builds its own local state (request_id, transcript, passages,
    timings dict) from scratch — nothing here is module-level mutable state,
    so back-to-back calls never leak into each other.
    """
    request_id = uuid.uuid4().hex[:12]
    timings_ms: Dict[str, float] = {}
    transcript, normalized_query, passages, answer_text = "", "", [], ""
    status, error_message = "pending", None

    try:
        t0 = time.perf_counter()
        transcript = transcribe(audio_path)
        timings_ms["stt"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        normalized_query = normalize(transcript)
        timings_ms["normalization"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        passages = hybrid_retrieve(normalized_query, top_k=TOP_K)
        timings_ms["retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        gate_passed = evidence_gate_check(normalized_query, passages)
        timings_ms["evidence_gate"] = round((time.perf_counter() - t0) * 1000, 2)

        if not gate_passed:
            status = "refused_no_evidence"
            answer_text = REFUSAL_NO_EVIDENCE
            return _render(request_id, transcript, passages, answer_text, status, error_message, timings_ms)

        t0 = time.perf_counter()
        answer_text, is_grounded = generate_answer(normalized_query, passages)
        timings_ms["generation"] = round((time.perf_counter() - t0) * 1000, 2)

        if not is_grounded:
            status = "refused_ungrounded"
            answer_text = REFUSAL_UNGROUNDED
            return _render(request_id, transcript, passages, answer_text, status, error_message, timings_ms)

        status = "answered"
        return _render(request_id, transcript, passages, answer_text, status, error_message, timings_ms)

    except STTError as exc:
        status, error_message = "error", str(exc)
        return _render(request_id, transcript, passages, "", status, error_message, timings_ms,
                        error=f"STT error: {exc}")

    except NormalizationError as exc:
        status, error_message = "error", str(exc)
        return _render(request_id, transcript, passages, "", status, error_message, timings_ms,
                        error=f"Normalization error: {exc}")

    except Exception as exc:  # noqa: BLE001 — top-level safety net so the UI never crashes
        status, error_message = "error", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return _render(request_id, transcript, passages, "", status, error_message, timings_ms,
                        error="Unexpected pipeline error - see server logs.")


def _render(
    request_id: str,
    transcript: str,
    passages: List[Dict[str, Any]],
    answer_text: str,
    status: str,
    error_message: Optional[str],
    timings_ms: Dict[str, float],
    error: Optional[str] = None,
) -> Tuple[str, str, str, str, str]:
    """Converts raw pipeline results into the 5 markdown strings the UI panels need."""

    transcript_md = transcript if transcript else _EMPTY_TRANSCRIPT

    if passages:
        rows = []
        for p in passages:
            preview = p["text"][:120] + ("…" if len(p["text"]) > 120 else "")
            rows.append(
                f"- **{p['passage_id']}** | score: `{p['score']:.3f}` | "
                f"source: `{p['source']}`\n  > {preview}"
            )
        evidence_md = "\n".join(rows)
    else:
        evidence_md = _EMPTY_EVIDENCE

    answer_md = f"**ERROR** - {error}" if error else (answer_text if answer_text else _EMPTY_ANSWER)

    status_md = f"`{status}`  ·  request `{request_id}`"
    if error_message:
        status_md += f"\n\n_{error_message}_"

    if timings_ms:
        total_ms = round(sum(timings_ms.values()), 2)
        lat_rows = [f"- **{name}**: {ms:.2f} ms" for name, ms in timings_ms.items()]
        lat_rows.append(f"- **total**: {total_ms:.2f} ms")
        latency_md = "\n".join(lat_rows)
    else:
        latency_md = _EMPTY_LATENCY

    return transcript_md, evidence_md, answer_md, status_md, latency_md


# ============================================================================
# 7. Gradio UI
# ============================================================================

custom_css = """
/* Force light-mode CSS variables even in dark mode */
:root, .dark, body, html, .gradio-container {
    --body-background-fill: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 100%) !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #bae6fd !important;
    --border-color-primary: #bae6fd !important;
    --border-color-secondary: #bae6fd !important;
    
    /* Text colors */
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --input-text-color: #0f172a !important;
    
    /* Input background */
    --input-background-fill: #ffffff !important;
}

/* Custom premium theme for Voice-RAG Observability Panel */
body, html, .gradio-container {
    background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 100%) !important;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Force text colors to be dark slate/black for high contrast */
.gradio-container * {
    color: #0f172a !important; /* Premium dark slate/black */
}

/* Card styling */
.header-card, .input-card, .output-card {
    background-color: #ffffff !important;
    border: 1.5px solid #bae6fd !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 0px !important;
}

.output-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    border-color: #7dd3fc !important;
}

/* Remove default Gradio block background/border highlight inside our card wrappers */
.header-card div, .input-card div, .output-card div {
    background: transparent !important;
    background-color: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Restore backgrounds for specific formatted elements */
code, pre, pre code, .output-card code, .output-card pre {
    background-color: #f1f5f9 !important;
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #bae6fd !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
}

/* Restore native styling for the audio player component controls */
.my-audio-player, .my-audio-player * {
    background-color: initial !important;
    background: initial !important;
    border-color: initial !important;
}

/* Primary Button Styling */
button.primary, .run-btn {
    background-color: #2563eb !important;
    background: #2563eb !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
}

button.primary:hover, .run-btn:hover {
    background-color: #1d4ed8 !important;
    background: #1d4ed8 !important;
    box-shadow: 0 6px 12px -1px rgba(37, 99, 235, 0.3) !important;
    transform: translateY(-1px);
}

/* Exception for primary button text and children, which should remain white */
button.primary, .run-btn, button.primary *, .run-btn * {
    color: #ffffff !important;
}

/* Space utilization: height distribution */
.observability-row {
    min-height: 620px !important;
    gap: 24px !important;
}

.observability-column {
    display: flex !important;
    flex-direction: column !important;
    gap: 24px !important;
    justify-content: stretch !important;
    height: 100% !important;
}

.output-card {
    flex: 1 1 auto !important;
}

/* Code block and status tag styling */
code, pre, pre code, .output-card code, .output-card pre {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #bae6fd !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 100%)",
    body_text_color="#0f172a",
    body_text_color_subdued="#334155",
    block_background_fill="#ffffff",
    block_label_text_color="#0f172a",
    block_title_text_color="#0f172a",
)

with gr.Blocks(theme=theme, title="Voice-RAG Observability Panel", css=custom_css) as demo:
    with gr.Group(elem_classes=["header-card"]):
        gr.Markdown(
            "# 🎙️ Voice-RAG Observability Panel\n"
            "Instrumented engineering control panel for a **Hindi** "
            "voice → hybrid retrieval → grounded generation pipeline. "
            "Every stage is timed and every intermediate artifact "
            "(transcript, retrieved evidence, grounding decision) is exposed below. "
            "Single-process control panel, not a chat app: no auth, no history, "
            "one request in → one fully-traced response out."
        )

    with gr.Group(elem_classes=["input-card"]):
        audio_in = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Ask a question (Hindi audio)",
            elem_classes=["my-audio-player"],
        )
        run_btn = gr.Button("▶ Run Pipeline", variant="primary", elem_classes=["run-btn"])

    with gr.Row(elem_classes=["observability-row"]):
        with gr.Column(elem_classes=["observability-column"]):
            with gr.Group(elem_classes=["output-card"]):
                gr.Markdown("### 📝 Transcript")
                transcript_out = gr.Markdown(value=_EMPTY_TRANSCRIPT)

            with gr.Group(elem_classes=["output-card"]):
                gr.Markdown("### 📚 Retrieved Evidence")
                evidence_out = gr.Markdown(value=_EMPTY_EVIDENCE)

            with gr.Group(elem_classes=["output-card"]):
                gr.Markdown("### 💬 Answer")
                answer_out = gr.Markdown(value=_EMPTY_ANSWER)

        with gr.Column(elem_classes=["observability-column"]):
            with gr.Group(elem_classes=["output-card"]):
                gr.Markdown("### ⚙️ Status")
                status_out = gr.Markdown(value=_EMPTY_STATUS)

            with gr.Group(elem_classes=["output-card"]):
                gr.Markdown("### ⏱️ Latency (per stage, ms)")
                latency_out = gr.Markdown(value=_EMPTY_LATENCY)

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_in],
        outputs=[transcript_out, evidence_out, answer_out, status_out, latency_out],
    )

if __name__ == "__main__":
    demo.launch()
