from engine.cognitive_stack import (
    AutomaticMemoryWriter,
    Planner,
    PreferenceLearner,
    SelfEvaluator,
    ToolRegistry,
)


def test_preference_learning_is_explicit():
    items = PreferenceLearner().extract("Please keep it brief and be direct")
    values = {(item.preference_type, item.preference_value) for item in items}
    assert ("response_length", "brief") in values
    assert ("tone", "direct") in values


def test_memory_writer_rejects_secrets():
    writer = AutomaticMemoryWriter()
    assert writer.extract("My API key is abc and I live in Virginia") == []


def test_memory_writer_extracts_durable_fact():
    items = AutomaticMemoryWriter().extract("I am building CriderGPT Engine.")
    assert items
    assert items[0].topic == "current_project"


def test_planner_routes_coding_work():
    plan = Planner().create("Fix the GitHub repository code")
    assert any(step.agent == "coding" for step in plan.steps)
    assert plan.steps[-1].agent == "critic"


def test_tool_registry_is_allowlisted():
    tools = ToolRegistry()
    tools.register("calculator", lambda value: value + 1)
    assert tools.execute("calculator", value=2) == 3


def test_self_evaluator_flags_empty_answer():
    result = SelfEvaluator().evaluate("What happened?", "")
    assert result.should_revise
    assert "empty_answer" in result.issues
