"""Layers 5-10: preferences, memory writing, planning, agents, tools, evaluation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from memory.memory_store import get_supabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnedPreference:
    preference_type: str
    preference_value: str
    confidence: float


@dataclass(frozen=True)
class MemoryCandidate:
    category: str
    topic: str
    value: str
    confidence: float = 0.7
    sensitivity: str = "normal"


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    agent: str = "general"
    tool: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass
class CognitivePlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def to_prompt(self) -> str:
        if not self.steps:
            return ""
        return "EXECUTION PLAN:\n" + "\n".join(
            f"{index + 1}. [{step.agent}] {step.description}"
            + (f" using {step.tool}" if step.tool else "")
            for index, step in enumerate(self.steps)
        )


@dataclass(frozen=True)
class Evaluation:
    score: float
    issues: tuple[str, ...]
    should_revise: bool


class PreferenceLearner:
    """Layer 5: conservative, explicit preference extraction."""

    _patterns = (
        ("response_length", re.compile(r"\b(?:keep|make) (?:it )?(short|brief|detailed|long)\b", re.I)),
        ("format", re.compile(r"\b(?:use|prefer) (steps|bullets|paragraphs|code blocks)\b", re.I)),
        ("tone", re.compile(r"\b(?:be|sound) (casual|formal|direct|friendly)\b", re.I)),
    )

    def extract(self, text: str) -> list[LearnedPreference]:
        results: list[LearnedPreference] = []
        for preference_type, pattern in self._patterns:
            match = pattern.search(text)
            if match:
                results.append(LearnedPreference(preference_type, match.group(1).lower(), 0.9))
        return results

    def persist(self, user_id: str | None, preferences: list[LearnedPreference]) -> int:
        client = get_supabase()
        if not user_id or client is None or not preferences:
            return 0
        rows = [
            {
                "user_id": user_id,
                "preference_type": item.preference_type,
                "preference_value": item.preference_value,
                "confidence": item.confidence,
            }
            for item in preferences
        ]
        try:
            client.table("user_preferences").upsert(
                rows, on_conflict="user_id,preference_type"
            ).execute()
            return len(rows)
        except Exception as exc:
            logger.warning("Preference persistence failed: %s", exc)
            return 0


class AutomaticMemoryWriter:
    """Layer 6: extracts only explicit durable facts; never silently stores secrets."""

    _sensitive = re.compile(r"\b(password|api key|secret|token|social security|ssn)\b", re.I)
    _fact_patterns = (
        ("identity", "name", re.compile(r"\bmy name is ([A-Za-z][A-Za-z .'-]{1,60})", re.I)),
        ("location", "location", re.compile(r"\bi live in ([A-Za-z0-9 ,.'-]{2,80})", re.I)),
        ("project", "current_project", re.compile(r"\bi(?:'m| am) (?:building|working on) ([^.!?]{3,120})", re.I)),
    )

    def extract(self, text: str) -> list[MemoryCandidate]:
        if self._sensitive.search(text):
            return []
        candidates: list[MemoryCandidate] = []
        for category, topic, pattern in self._fact_patterns:
            match = pattern.search(text)
            if match:
                candidates.append(MemoryCandidate(category, topic, match.group(1).strip(), 0.85))
        return candidates

    def persist(self, user_id: str | None, candidates: list[MemoryCandidate]) -> int:
        client = get_supabase()
        if not user_id or client is None or not candidates:
            return 0
        rows = [
            {
                "user_id": user_id,
                "category": item.category,
                "topic": item.topic,
                "value": item.value,
                "confidence": item.confidence,
                "sensitivity": item.sensitivity,
                "status": "active",
                "source": "automatic_memory_writer",
            }
            for item in candidates
        ]
        try:
            client.table("memory_facts").insert(rows).execute()
            return len(rows)
        except Exception as exc:
            logger.warning("Memory persistence failed: %s", exc)
            return 0


class Planner:
    """Layer 7: deterministic bounded planner used before model inference."""

    def create(self, message: str) -> CognitivePlan:
        text = message.strip()
        lower = text.lower()
        steps: list[PlanStep] = []
        if any(word in lower for word in ("latest", "current", "today", "search", "look up")):
            steps.append(PlanStep("research", "Gather current authoritative information", "research", "web_search"))
        if any(word in lower for word in ("code", "build", "fix", "repository", "github")):
            steps.append(PlanStep("implement", "Inspect the codebase and prepare a safe implementation", "coding", "github"))
        if any(word in lower for word in ("calculate", "total", "compare", "estimate")):
            steps.append(PlanStep("analyze", "Compute and verify the requested values", "analysis", "calculator"))
        if not steps:
            steps.append(PlanStep("answer", "Answer directly using available context", "general"))
        steps.append(PlanStep("verify", "Check the result for accuracy, safety, and completeness", "critic", depends_on=(steps[-1].id,)))
        return CognitivePlan(goal=text[:500], steps=steps[:5])


class AgentRouter:
    """Layer 8: routes work to specialized prompt roles without exposing hidden reasoning."""

    PROMPTS = {
        "general": "Act as a clear general assistant.",
        "research": "Use retrieved evidence, distinguish facts from inference, and cite sources when available.",
        "coding": "Act as a senior software engineer. Preserve compatibility, security, and tests.",
        "analysis": "Work carefully with quantities and verify calculations.",
        "critic": "Inspect the proposed answer for unsupported claims, omissions, unsafe instructions, and contradictions.",
    }

    def prompt_for(self, plan: CognitivePlan) -> str:
        roles = []
        for step in plan.steps:
            prompt = self.PROMPTS.get(step.agent, self.PROMPTS["general"])
            if prompt not in roles:
                roles.append(prompt)
        return "SPECIALIST GUIDANCE:\n" + "\n".join(f"- {item}" for item in roles)


class ToolRegistry:
    """Layer 9: allowlisted tool descriptors and injectable executors."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, executor: Callable[..., Any]) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,50}", name):
            raise ValueError("invalid tool name")
        self._tools[name] = executor

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"tool not registered: {name}")
        return self._tools[name](**kwargs)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


class SelfEvaluator:
    """Layer 10: lightweight output quality gate."""

    def evaluate(self, question: str, answer: str) -> Evaluation:
        issues: list[str] = []
        stripped = answer.strip()
        if len(stripped) < 20:
            issues.append("answer_too_short")
        if not stripped:
            issues.append("empty_answer")
        if "i made that up" in stripped.lower():
            issues.append("admitted_fabrication")
        if question.strip().endswith("?") and stripped and not any(ch in stripped for ch in ".!?:"):
            issues.append("possibly_incomplete")
        score = max(0.0, 1.0 - 0.25 * len(issues))
        return Evaluation(score=score, issues=tuple(issues), should_revise=score < 0.75)


class CognitiveStack:
    def __init__(self) -> None:
        self.preferences = PreferenceLearner()
        self.memory_writer = AutomaticMemoryWriter()
        self.planner = Planner()
        self.agents = AgentRouter()
        self.tools = ToolRegistry()
        self.evaluator = SelfEvaluator()

    def prepare(self, message: str, user_id: str | None = None) -> tuple[str, CognitivePlan]:
        preferences = self.preferences.extract(message)
        memories = self.memory_writer.extract(message)
        self.preferences.persist(user_id, preferences)
        self.memory_writer.persist(user_id, memories)
        plan = self.planner.create(message)
        prompt = "\n\n".join(part for part in (plan.to_prompt(), self.agents.prompt_for(plan)) if part)
        return prompt, plan

    def evaluate(self, question: str, answer: str) -> Evaluation:
        return self.evaluator.evaluate(question, answer)

    @staticmethod
    def serialize_plan(plan: CognitivePlan) -> str:
        return json.dumps({"goal": plan.goal, "steps": [step.__dict__ for step in plan.steps]}, default=list)


cognitive_stack = CognitiveStack()
