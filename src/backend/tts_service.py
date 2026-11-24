"""GPU-only local Qwen3-TTS service used by the optional Compose profile."""

from contextlib import asynccontextmanager
from io import BytesIO
import logging
import re
from threading import Lock

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "Aiden"
INSTRUCTION = "Speak in clear, neutral-professional English."
CHUNK_CHARACTERS = 150
logger = logging.getLogger("qwen_tts_service")

_model = None
_generation_lock = Lock()


def _load_model():
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    try:
        _model = _load_model()
    except Exception as exc:
        logger.error("TTS model preload failed (%s)", type(exc).__name__)
        raise
    logger.info("TTS model preloaded: %s", MODEL_ID)
    yield
    _model = None


app = FastAPI(title="Local Qwen3-TTS", lifespan=lifespan)


class SynthesisRequest(BaseModel):
    text: str


def _split_text(text: str) -> list[str]:
    """Split at sentence boundaries, with a whitespace fallback for long sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        pieces: list[str] = []
        piece = ""
        for word in words:
            if piece and len(piece) + len(word) + 1 > CHUNK_CHARACTERS:
                pieces.append(piece)
                piece = word
            else:
                piece = f"{piece} {word}".strip()
        if piece:
            pieces.append(piece)

        for part in pieces:
            if current and len(current) + len(part) + 1 > CHUNK_CHARACTERS:
                chunks.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def _generate_wav(text: str) -> bytes:
    if _model is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")
    chunks = _split_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No answer text to synthesize.")

    audio_chunks = []
    sample_rate = None
    for chunk in chunks:
        waveforms, rate = _model.generate_custom_voice(
            text=chunk,
            language="English",
            speaker=SPEAKER,
            instruct=INSTRUCTION,
        )
        if sample_rate is not None and rate != sample_rate:
            raise RuntimeError("The model returned inconsistent sample rates.")
        sample_rate = rate
        audio_chunks.append(np.asarray(waveforms[0]).reshape(-1))

    output = BytesIO()
    sf.write(output, np.concatenate(audio_chunks), sample_rate, format="WAV")
    return output.getvalue()


@app.get("/healthz")
def healthcheck():
    return {"ready": _model is not None}


@app.post("/synthesize")
def synthesize(request: SynthesisRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="No answer text to synthesize.")
    try:
        # A single lock deliberately serializes inference on the one GPU.
        with _generation_lock:
            audio = _generate_wav(request.text)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("TTS generation failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Speech generation failed. Please retry.",
        ) from exc
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )
