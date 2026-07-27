from engine.inference import _select_local_model


def test_cloud_gateway_model_falls_back_to_local(monkeypatch):
    monkeypatch.setattr("engine.inference.settings.ollama_model", "llama3.1:8b")
    assert _select_local_model("google/gemini-3-flash-preview") == "llama3.1:8b"


def test_local_ollama_model_is_preserved(monkeypatch):
    monkeypatch.setattr("engine.inference.settings.ollama_model", "llama3.1:8b")
    assert _select_local_model("qwen2.5:7b") == "qwen2.5:7b"
