from types import SimpleNamespace

import engine.inference as inference


def test_cloud_gateway_model_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(inference, "settings", SimpleNamespace(ollama_model="llama3.1:8b"))
    assert inference._select_local_model("google/gemini-3-flash-preview") == "llama3.1:8b"


def test_local_ollama_model_is_preserved(monkeypatch):
    monkeypatch.setattr(inference, "settings", SimpleNamespace(ollama_model="llama3.1:8b"))
    assert inference._select_local_model("qwen2.5:7b") == "qwen2.5:7b"
