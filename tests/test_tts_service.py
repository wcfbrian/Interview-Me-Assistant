from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from src.backend import tts_service
from src.backend.tts_service import _load_model


def test_model_load_uses_flash_attention_2(monkeypatch):
    load_options = {}

    class FakeQwen3TTSModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            load_options["model_id"] = model_id
            load_options.update(kwargs)
            return object()

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel),
    )

    _load_model()

    assert load_options == {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "device_map": "cuda:0",
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }


def test_generate_wav_uses_the_prepared_voice_clone_prompt(monkeypatch):
    clone_prompt = object()

    class FakeModel:
        def generate_voice_clone(self, **options):
            assert options == {
                "text": "Hello there.",
                "language": "English",
                "voice_clone_prompt": clone_prompt,
            }
            return [np.zeros(240, dtype=np.float32)], 24000

    monkeypatch.setattr(tts_service, "_model", FakeModel())
    monkeypatch.setattr(tts_service, "_voice_clone_prompt", clone_prompt)

    audio = tts_service._generate_wav("Hello there.")

    assert audio.startswith(b"RIFF")
    assert audio[8:12] == b"WAVE"


def test_voice_clone_prompt_is_saved_and_reused(monkeypatch, tmp_path):
    @dataclass
    class FakePrompt:
        ref_code: torch.Tensor
        ref_spk_embedding: torch.Tensor
        x_vector_only_mode: bool
        icl_mode: bool
        ref_text: str

    qwen_module = ModuleType("qwen_tts")
    qwen_module.__path__ = []
    inference_module = ModuleType("qwen_tts.inference")
    inference_module.__path__ = []
    prompt_module = ModuleType("qwen_tts.inference.qwen3_tts_model")
    prompt_module.VoiceClonePromptItem = FakePrompt
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_module)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference", inference_module)
    monkeypatch.setitem(
        sys.modules, "qwen_tts.inference.qwen3_tts_model", prompt_module
    )

    audio_path = tmp_path / "my_voice.wav"
    text_path = tmp_path / "my_voice.txt"
    cache_path = tmp_path / "prompt.safetensors"
    audio_path.write_bytes(b"reference audio")
    text_path.write_text("Reference transcript.", encoding="utf-8")
    monkeypatch.setattr(tts_service, "REFERENCE_AUDIO", audio_path)
    monkeypatch.setattr(tts_service, "REFERENCE_TEXT", text_path)
    monkeypatch.setattr(tts_service, "VOICE_CLONE_CACHE", cache_path)

    class FakeModel:
        device = "cpu"
        prompt_creations = 0

        def create_voice_clone_prompt(self, **options):
            self.prompt_creations += 1
            assert options == {
                "ref_audio": str(audio_path),
                "ref_text": text_path.read_text(encoding="utf-8").strip(),
                "x_vector_only_mode": False,
            }
            return [
                FakePrompt(
                    ref_code=torch.tensor([[1, 2]], dtype=torch.long),
                    ref_spk_embedding=torch.tensor([0.25, 0.5]),
                    x_vector_only_mode=False,
                    icl_mode=True,
                    ref_text=options["ref_text"],
                )
            ]

    model = FakeModel()
    created_prompt = tts_service._load_or_create_voice_clone_prompt(model)[0]
    cached_prompt = tts_service._load_or_create_voice_clone_prompt(model)[0]

    assert cache_path.is_file()
    assert model.prompt_creations == 1
    assert torch.equal(cached_prompt.ref_code, created_prompt.ref_code)
    assert torch.equal(
        cached_prompt.ref_spk_embedding, created_prompt.ref_spk_embedding
    )
    assert cached_prompt.ref_text == "Reference transcript."

    text_path.write_text("Updated reference transcript.", encoding="utf-8")
    updated_prompt = tts_service._load_or_create_voice_clone_prompt(model)[0]

    assert model.prompt_creations == 2
    assert updated_prompt.ref_text == "Updated reference transcript."
