"""GPU-only local Qwen3-TTS service used by the optional Compose profile."""

from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import tempfile
from threading import Lock

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE_AUDIO = Path(
    os.getenv("QWEN_TTS_REFERENCE_AUDIO", "/app/data/my_voice.wav")
)
REFERENCE_TEXT = Path(os.getenv("QWEN_TTS_REFERENCE_TEXT", "/app/data/my_voice.txt"))
VOICE_CLONE_CACHE = Path(
    os.getenv("QWEN_TTS_VOICE_CLONE_CACHE", "/voice-clone/prompt.safetensors")
)
CHUNK_CHARACTERS = 600
logger = logging.getLogger("qwen_tts_service")

_model = None
_voice_clone_prompt = None
_generation_lock = Lock()


def _load_model():
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def _reference_fingerprint(ref_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL_ID.encode("utf-8"))
    digest.update(b"\0")
    digest.update(REFERENCE_AUDIO.read_bytes())
    digest.update(b"\0")
    digest.update(ref_text.encode("utf-8"))
    return digest.hexdigest()


def _load_or_create_voice_clone_prompt(model):
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    from safetensors import SafetensorError, safe_open
    from safetensors.torch import save_file

    if not REFERENCE_AUDIO.is_file():
        raise FileNotFoundError(
            f"Voice reference audio not found: {REFERENCE_AUDIO}"
        )
    if not REFERENCE_TEXT.is_file():
        raise FileNotFoundError(f"Voice reference transcript not found: {REFERENCE_TEXT}")
    ref_text = REFERENCE_TEXT.read_text(encoding="utf-8").strip()
    if not ref_text:
        raise ValueError(
            f"Voice reference transcript is empty: {REFERENCE_TEXT}"
        )

    fingerprint = _reference_fingerprint(ref_text)
    if VOICE_CLONE_CACHE.is_file():
        try:
            with safe_open(
                str(VOICE_CLONE_CACHE), framework="pt", device=str(model.device)
            ) as cache:
                metadata = cache.metadata() or {}
                if (
                    metadata.get("model_id") == MODEL_ID
                    and metadata.get("fingerprint") == fingerprint
                ):
                    prompt = VoiceClonePromptItem(
                        ref_code=cache.get_tensor("ref_code"),
                        ref_spk_embedding=cache.get_tensor("ref_spk_embedding"),
                        x_vector_only_mode=False,
                        icl_mode=True,
                        ref_text=ref_text,
                    )
                    logger.info(
                        "Loaded cached voice clone prompt: %s", VOICE_CLONE_CACHE
                    )
                    return [prompt]
        except (SafetensorError, KeyError):
            logger.warning("Cached voice clone prompt is invalid; rebuilding it.")

    prompts = model.create_voice_clone_prompt(
        ref_audio=str(REFERENCE_AUDIO),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    prompt = prompts[0]
    if prompt.ref_code is None:
        raise RuntimeError("Qwen3-TTS did not create reference speech codes.")

    VOICE_CLONE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=VOICE_CLONE_CACHE.parent,
            prefix=f".{VOICE_CLONE_CACHE.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        save_file(
            {
                "ref_code": prompt.ref_code.detach().cpu().contiguous(),
                "ref_spk_embedding": prompt.ref_spk_embedding.detach()
                .cpu()
                .contiguous(),
            },
            str(temporary_path),
            metadata={"model_id": MODEL_ID, "fingerprint": fingerprint},
        )
        os.replace(temporary_path, VOICE_CLONE_CACHE)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    logger.info("Created and cached voice clone prompt: %s", VOICE_CLONE_CACHE)
    return prompts


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _voice_clone_prompt
    try:
        _model = _load_model()
        _voice_clone_prompt = _load_or_create_voice_clone_prompt(_model)
    except Exception as exc:
        logger.error(
            "TTS model or voice clone preload failed (%s)", type(exc).__name__
        )
        raise
    logger.info("TTS model and voice clone preloaded: %s", MODEL_ID)
    yield
    _model = None
    _voice_clone_prompt = None


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
    if _model is None or _voice_clone_prompt is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")
    chunks = _split_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No answer text to synthesize.")

    audio_chunks = []
    sample_rate = None
    for chunk in chunks:
        waveforms, rate = _model.generate_voice_clone(
            text=chunk,
            language="English",
            voice_clone_prompt=_voice_clone_prompt,
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
    return {"ready": _model is not None and _voice_clone_prompt is not None}


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
