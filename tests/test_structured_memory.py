from datetime import datetime, timezone

from memory.structured_memory import MemoryFact


def make_fact(**overrides):
    data = {
        "id": "fact-1",
        "user_id": "user-1",
        "subject": "Minecraft server",
        "predicate": "available RAM is",
        "value": "16 GB",
        "category": "server",
        "status": "active",
        "sensitivity": "private",
        "confidence": 0.95,
        "valid_from": None,
        "valid_until": None,
        "source": "user_statement",
        "source_date": None,
        "supersedes_id": None,
        "last_verified_at": None,
        "review_after": None,
        "created_at": "2026-07-27T00:00:00Z",
    }
    data.update(overrides)
    return MemoryFact.from_row(data)


def test_active_fact_is_current():
    assert make_fact().is_current(datetime.now(timezone.utc)) is True


def test_superseded_fact_is_not_current():
    assert make_fact(status="superseded").is_current() is False


def test_expired_fact_is_not_current():
    assert make_fact(valid_until="2020-01-01T00:00:00Z").is_current() is False


def test_prompt_line_contains_confidence_and_status():
    line = make_fact().to_prompt_line()
    assert "Minecraft server available RAM is 16 GB" in line
    assert "confidence=0.95" in line
    assert "status=active" in line
