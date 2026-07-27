from memory.semantic_rag import RagChunk
from scripts.ingest_rag_document import chunk_text


def test_chunk_text_preserves_content_with_overlap():
    text = "First paragraph. " * 100 + "\n\n" + "Second paragraph. " * 100
    chunks = chunk_text(text, size=300, overlap=40)
    assert len(chunks) > 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_rag_chunk_prompt_includes_source():
    chunk = RagChunk(
        source_id="abc",
        title="CriderShield notes",
        content="The DNS service uses dns2.",
        source_type="project",
        source_uri="github://cridershield",
        similarity=0.92,
    )
    rendered = chunk.to_prompt_line()
    assert "CriderShield notes" in rendered
    assert "github://cridershield" in rendered
    assert "dns2" in rendered
