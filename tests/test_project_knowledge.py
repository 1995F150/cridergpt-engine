from memory.project_knowledge import ProjectKnowledge, _score, _terms


def test_project_knowledge_prompt_line():
    item = ProjectKnowledge(
        id="1",
        user_id="user-1",
        project_key="cridershield",
        project_name="CriderShield",
        category="deployment",
        title="System service",
        content="Run as a systemd service instead of Docker.",
        status="active",
        priority=90,
    )
    line = item.to_prompt_line()
    assert "CriderShield" in line
    assert "systemd" in line
    assert "active" in line


def test_project_name_matches_rank_higher_than_body_only():
    terms = _terms("CriderShield installer")
    project_match = ProjectKnowledge(
        id="1",
        user_id="u",
        project_key="cridershield",
        project_name="CriderShield",
        category="general",
        title="Overview",
        content="DNS filtering",
        status="active",
        priority=50,
    )
    body_match = ProjectKnowledge(
        id="2",
        user_id="u",
        project_key="other",
        project_name="Other",
        category="general",
        title="Notes",
        content="The CriderShield installer is mentioned here.",
        status="active",
        priority=50,
    )
    assert _score(project_match, terms) > _score(body_match, terms)
