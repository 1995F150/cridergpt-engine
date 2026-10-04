#!/usr/bin/env python3
"""Convert a ChatGPT-style conversation export into CriderGPT 2.0 training JSONL.

The raw export stays local. This script follows each conversation's active branch,
keeps user/assistant order, removes duplicates, skips empty/tool/system messages,
redacts obvious secrets, and writes CriderGPT's native role-token format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
EOS_TAG = "<|eos|>"

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_ -]?key|secret|password|passwd|token)\s*[:=]\s*[^\s,;]{8,}"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
]


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text.strip()


def message_text(message: dict) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    chunks = [part for part in parts if isinstance(part, str)]
    return redact("\n".join(chunks))


def active_path(conv: dict) -> list[dict]:
    mapping = conv.get("mapping") or {}
    current = conv.get("current_node")

    if current and current in mapping:
        ids: list[str] = []
        seen: set[str] = set()
        node_id = current
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            ids.append(node_id)
            node_id = mapping[node_id].get("parent")
        ids.reverse()
        return [mapping[node_id] for node_id in ids]

    # Fallback for exports without current_node: stable chronological order.
    nodes = list(mapping.values())
    nodes.sort(key=lambda n: ((n.get("message") or {}).get("create_time") or 0))
    return nodes


def normalize_conversation(conv: dict, max_chars: int) -> str | None:
    turns: list[tuple[str, str]] = []
    for node in active_path(conv):
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        role = (message.get("author") or {}).get("role")
        if role not in {"user", "assistant"}:
            continue
        text = message_text(message)
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
        if turns and turns[-1][0] == role:
            # Merge adjacent same-role messages instead of inventing a reply pair.
            turns[-1] = (role, turns[-1][1] + "\n" + text)
        else:
            turns.append((role, text))

    # Require at least one complete user -> assistant exchange.
    if not any(turns[i][0] == "user" and turns[i + 1][0] == "assistant" for i in range(len(turns) - 1)):
        return None

    # Remove leading assistant chatter and dangling final user prompt.
    while turns and turns[0][0] != "user":
        turns.pop(0)
    if turns and turns[-1][0] == "user":
        turns.pop()
    if len(turns) < 2:
        return None

    rendered: list[str] = []
    for role, text in turns:
        tag = USER_TAG if role == "user" else ASSISTANT_TAG
        rendered.append(f"{tag}\n{text}")
    return "\n".join(rendered) + f"\n{EOS_TAG}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ChatGPT-style conversations JSON export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/cridergpt2/conversations.jsonl"),
    )
    parser.add_argument("--max-message-chars", type=int, default=12000)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Expected the export root to be a JSON list of conversations.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    accepted = 0
    rejected = 0

    with args.output.open("w", encoding="utf-8") as out:
        for conv in data:
            if not isinstance(conv, dict):
                rejected += 1
                continue
            text = normalize_conversation(conv, args.max_message_chars)
            if not text:
                rejected += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            accepted += 1

    print(f"Accepted conversations: {accepted:,}")
    print(f"Rejected/duplicate conversations: {rejected:,}")
    print(f"Output: {args.output}")
    print("Review the generated JSONL before training; raw exports may contain personal data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
