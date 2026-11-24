"""Local voice input and output adapters; no audio or speech text is persisted here."""

from functools import lru_cache
from io import BytesIO
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.backend.config import settings


class VoiceInputError(RuntimeError):
    """A safe, user-displayable voice transcription error."""


class VoiceOutputError(RuntimeError):
    """A safe, user-displayable local speech-generation error."""


@lru_cache(maxsize=1)
def _whisper_model():
    """Load the small English model only on the first voice-input request."""
    from faster_whisper import WhisperModel

    settings.whisper_cache_dir.mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        download_root=str(settings.whisper_cache_dir),
    )


def transcribe_audio(audio: bytes) -> str:
    """Transcribe browser-recorded audio in memory, in English, on CPU."""
    if not audio:
        raise VoiceInputError("No audio was recorded. Please record again.")
    if len(audio) > settings.max_voice_audio_bytes:
        raise VoiceInputError(
            "This recording is too large to process. Please record a shorter clip."
        )

    try:
        segments, _ = _whisper_model().transcribe(
            BytesIO(audio),
            language="en",
            task="transcribe",
            vad_filter=True,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        raise VoiceInputError(
            "Transcription failed. Please record again and retry."
        ) from exc

    if not transcript:
        raise VoiceInputError(
            "No English speech was detected. Please record again and retry."
        )
    return transcript


_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_CODE_FENCE = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]*>")


def clean_answer_for_speech(answer: str) -> str:
    """Remove common Markdown presentation markers while preserving answer text."""
    text = _CODE_FENCE.sub(r"\1", answer)
    text = _MARKDOWN_IMAGE.sub(r"\1", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"~~(.*?)~~", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)(\*|_)(.+?)\1(?!\w)", r"\2", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def synthesize_speech(answer: str) -> bytes:
    """Request a WAV response from the optional, local Qwen TTS service."""
    text = clean_answer_for_speech(answer)
    if not text:
        raise VoiceOutputError("There is no answer text available to speak.")

    body = json.dumps({"text": text}).encode("utf-8")
    try:
        request = Request(
            f"{settings.tts_base_url.rstrip('/')}/synthesize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=settings.tts_timeout_seconds) as response:
            audio = response.read()
    except HTTPError as exc:
        raise VoiceOutputError(
            "Speech generation failed. Please retry voice output."
        ) from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise VoiceOutputError(
            "Local voice output is unavailable. "
            "Check that the GPU TTS service is running."
        ) from exc

    if (
        not audio
        or len(audio) < 12
        or not audio.startswith(b"RIFF")
        or audio[8:12] != b"WAVE"
    ):
        raise VoiceOutputError(
            "The GPU TTS service returned invalid audio. Please retry voice output."
        )
    return audio
