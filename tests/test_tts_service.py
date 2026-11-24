import sys
from types import SimpleNamespace

import torch

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
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "device_map": "cuda:0",
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
