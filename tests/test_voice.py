from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.backend.config import Settings, settings
from src.backend.voice import (
    VoiceInputError,
    clean_answer_for_speech,
    transcribe_audio,
)


def test_transcription_is_english_and_returns_the_joined_transcript(monkeypatch):
    class FakeWhisper:
        def transcribe(self, audio, **options):
            assert audio.read() == b"recorded audio"
            assert options == {
                "language": "en",
                "task": "transcribe",
                "vad_filter": True,
            }
            return iter(
                [
                    SimpleNamespace(text=" Tell me "),
                    SimpleNamespace(text="about this project. "),
                ]
            ), object()

    monkeypatch.setattr("src.backend.voice._whisper_model", lambda: FakeWhisper())

    assert transcribe_audio(b"recorded audio") == "Tell me about this project."


def test_empty_audio_is_rejected_without_loading_a_model(monkeypatch):
    def unexpected_model_load():
        pytest.fail("Empty audio must be rejected before model initialization")

    monkeypatch.setattr("src.backend.voice._whisper_model", unexpected_model_load)

    with pytest.raises(VoiceInputError, match="No audio was recorded"):
        transcribe_audio(b"")


def test_oversized_audio_is_rejected_before_model_loading(monkeypatch):
    monkeypatch.setattr(
        "src.backend.voice.settings",
        replace(settings, max_voice_audio_bytes=3),
    )
    monkeypatch.setattr(
        "src.backend.voice._whisper_model",
        lambda: pytest.fail("Oversized audio must be rejected before model loading"),
    )

    with pytest.raises(VoiceInputError, match="too large"):
        transcribe_audio(b"four")


def test_empty_transcription_is_not_submitted(monkeypatch):
    class FakeWhisper:
        def transcribe(self, *_args, **_kwargs):
            return iter([]), object()

    monkeypatch.setattr("src.backend.voice._whisper_model", lambda: FakeWhisper())

    with pytest.raises(VoiceInputError, match="No English speech"):
        transcribe_audio(b"recorded audio")


def test_tts_endpoint_rejects_non_local_urls():
    with pytest.raises(ValueError, match="supported local HTTP endpoint"):
        Settings(tts_base_url="https://tts.example.test")


def test_answer_cleaner_preserves_words_and_removes_markdown():
    answer = (
        "## Experience\n\n"
        "- I built **a grounded assistant** with [LangGraph](https://example.test).\n"
        "Run `uv sync`."
    )

    assert clean_answer_for_speech(answer) == (
        "Experience I built a grounded assistant with LangGraph. Run uv sync."
    )
