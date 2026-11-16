"""MaskGuard AI - an LLM agent (Claude or Groq) that analyses the detection logs.

It answers natural-language questions ("how was compliance this morning?"),
spots trends, visually double-checks violation snapshots, and writes reports.

Usage:
    python -m src.agent                       # interactive chat in the terminal
    python -m src.agent "Write today's compliance report"

Requires ANTHROPIC_API_KEY (Claude) or GROQ_API_KEY (Groq, e.g. openai/gpt-oss-120b) in .env.
Set LLM_PROVIDER=groq|claude to choose when both are present.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from config import CLAUDE_MODEL, GROQ_MODEL, REPORTS_DIR, SNAPSHOTS_DIR
from src import database

SYSTEM_PROMPT = """You are MaskGuard AI, the analyst for a real-time face mask detection system.
A CNN (MobileNetV2) classifies every detected face as wearing a mask or not; each sampled
detection is logged with a timestamp, label, confidence and, for violations, a snapshot.

Use your tools to ground every answer in the logged data - never invent numbers. When asked
for a report, gather the summary, the hourly trend and recent violations, then save a concise
Markdown report with save_report. If a snapshot is relevant (e.g. checking a low-confidence
violation for a false positive), inspect it. Keep answers short, specific and actionable:
highlight peak violation hours, compliance rate and practical recommendations.
An auto-monitoring agent also raises alerts (see get_recent_alerts). Answers may be read
aloud by a voice assistant, so lead with the key point in plain sentences.
Today is {today}."""

TOOLS = [
    {
        "name": "get_compliance_summary",
        "description": "Totals, violation count, compliance rate (%) and per-label counts for a time window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "number", "description": "Look-back window in hours. Omit for all time."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_hourly_trend",
        "description": "Per-hour totals and violations for a time window - use it to find peak violation hours.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "number", "description": "Look-back window in hours. Omit for all time."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_recent_detections",
        "description": "Most recent detection events (newest first), optionally violations only. "
                       "Rows include id, ts, source, label, confidence and snapshot path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max rows (1-200).", "minimum": 1, "maximum": 200},
                "violations_only": {"type": "boolean"},
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_snapshot",
        "description": "Return a saved violation snapshot image so you can visually verify it "
                       "(e.g. detect false positives). Pass a snapshot path from get_recent_detections.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_recent_alerts",
        "description": "Alerts raised by the auto-monitoring agent (low compliance, persistent "
                       "violations, automatic summaries), newest first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max rows (1-100).", "minimum": 1, "maximum": 100},
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_model_metrics",
        "description": "Validation metrics of the trained mask classifier (precision, recall, F1, confusion matrix).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "save_report",
        "description": "Save a Markdown report to the reports folder. Returns the file path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "markdown": {"type": "string", "description": "Full report body in Markdown."},
            },
            "required": ["title", "markdown"],
            "additionalProperties": False,
        },
    },
]

# Models that accept server-side refusal fallbacks ("default" mode).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def _inspect_snapshot(path: str):
    p = Path(path).resolve()
    if SNAPSHOTS_DIR.resolve() not in p.parents or not p.is_file():
        return "Error: path must be an existing file inside the snapshots folder."
    data = base64.standard_b64encode(p.read_bytes()).decode()
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
        {"type": "text", "text": f"Snapshot {p.name}"},
    ]


def _save_report(title: str, markdown: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")[:50]
    path = REPORTS_DIR / f"{datetime.now():%Y%m%d_%H%M}_{slug}.md"
    path.write_text(f"# {title}\n\n{markdown}\n", encoding="utf-8")
    return str(path)


def _model_metrics() -> str:
    path = REPORTS_DIR / "metrics.json"
    return path.read_text() if path.exists() else "No metrics found - the model has not been trained yet."


def run_tool(name: str, args: dict):
    if name == "get_compliance_summary":
        return json.dumps(database.summary(args.get("hours")))
    if name == "get_hourly_trend":
        return json.dumps(database.hourly(args.get("hours")))
    if name == "get_recent_detections":
        limit = max(1, min(int(args.get("limit", 20)), 200))
        return json.dumps(database.recent(limit, bool(args.get("violations_only"))))
    if name == "inspect_snapshot":
        return _inspect_snapshot(args["path"])
    if name == "get_recent_alerts":
        return json.dumps(database.recent_alerts(max(1, min(int(args.get("limit", 20)), 100))))
    if name == "get_model_metrics":
        return _model_metrics()
    if name == "save_report":
        return f"Saved report to {_save_report(args['title'], args['markdown'])}"
    raise ValueError(f"Unknown tool: {name}")


def provider() -> str | None:
    """Which LLM backend to use: LLM_PROVIDER, else whichever API key is configured."""
    choice = os.getenv("LLM_PROVIDER", "").lower()
    if choice in {"groq", "claude"}:
        return choice
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return None


def llm_available() -> bool:
    p = provider()
    if p == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if p == "claude":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    return False


def MaskGuardAgent(max_steps: int = 12):
    """Create the agent for the configured provider (Claude or Groq)."""
    if provider() == "groq":
        return GroqAgent(max_steps=max_steps)
    return ClaudeAgent(max_steps=max_steps)


def _system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=datetime.now().strftime("%A %d %B %Y, %H:%M"))


class ClaudeAgent:
    """Multi-turn Claude agent with a manual tool-use loop."""

    def __init__(self, model: str = CLAUDE_MODEL, max_steps: int = 12):
        self.client = anthropic.Anthropic()
        self.model = model
        self.label = f"Claude ({model})"
        self.max_steps = max_steps
        self.messages: list[dict] = []
        self.system = _system_prompt()

    def _create(self):
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=TOOLS,
            messages=self.messages,
            output_config={"effort": "medium"},
        )
        if self.model in _FALLBACK_MODELS:
            # Re-run on Anthropic's recommended model if a safety classifier declines.
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.beta.messages.create(**kwargs)

    def chat(self, user_message: str, on_tool=None) -> str:
        """Send a message, run tools until Claude is done, return the final text.

        on_tool(name, input) is an optional callback for showing progress in a UI.
        """
        self.messages.append({"role": "user", "content": user_message})
        for _ in range(self.max_steps):
            response = self._create()
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                return "Sorry - I can't help with that request."
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n_(response truncated)_"
                return text or "(no response)"

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if on_tool:
                    on_tool(block.name, block.input)
                try:
                    content, is_error = run_tool(block.name, block.input), False
                except Exception as exc:  # report tool failures back to Claude
                    content, is_error = f"Error: {exc}", True
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": content, "is_error": is_error})
            self.messages.append({"role": "user", "content": results})
        return "Stopped: too many tool steps without a final answer."

    def reset(self) -> None:
        self.messages = []


# Groq uses the OpenAI-style function-calling format.
GROQ_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


def _describe_snapshot_text(path: str) -> str:
    """Text stand-in for inspect_snapshot on models without vision."""
    p = Path(path).resolve()
    if SNAPSHOTS_DIR.resolve() not in p.parents or not p.is_file():
        return "Error: path must be an existing file inside the snapshots folder."
    rows = [r for r in database.recent(500, violations_only=True)
            if r["snapshot"] and Path(r["snapshot"]).resolve() == p]
    return json.dumps({
        "note": "This model cannot view images. Judge likely false positives from the CNN "
                "confidence: violations under ~70% confidence are worth a manual look.",
        "file": p.name,
        "detections_in_snapshot": [f"{r['ts']}: {r['label']} ({r['confidence']:.0%})" for r in rows],
    })


class GroqAgent:
    """Multi-turn agent on Groq (e.g. openai/gpt-oss-120b) with a manual tool-call loop."""

    def __init__(self, model: str | None = None, max_steps: int = 12):
        import groq
        self.client = groq.Groq()
        self.model = model or GROQ_MODEL
        self.label = f"Groq ({self.model})"
        self.max_steps = max_steps
        self.messages: list[dict] = [{"role": "system", "content": _system_prompt()}]

    def chat(self, user_message: str, on_tool=None) -> str:
        self.messages.append({"role": "user", "content": user_message})
        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=GROQ_TOOLS,
                tool_choice="auto", max_completion_tokens=8192,
            )
            choice = response.choices[0]
            msg = choice.message
            calls = msg.tool_calls or []
            entry = {"role": "assistant", "content": msg.content or ""}
            if calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls
                ]
            self.messages.append(entry)
            if not calls:
                text = (msg.content or "").strip()
                if choice.finish_reason == "length":
                    text += "\n\n_(response truncated)_"
                return text or "(no response)"

            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if on_tool:
                        on_tool(c.function.name, args)
                    if c.function.name == "inspect_snapshot":
                        content = _describe_snapshot_text(args["path"])
                    else:
                        content = run_tool(c.function.name, args)
                except Exception as exc:  # report tool failures back to the model
                    content = f"Error: {exc}"
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": content})
        return "Stopped: too many tool steps without a final answer."

    def reset(self) -> None:
        self.messages = self.messages[:1]


def describe_api_error(exc: Exception) -> str:
    import groq
    for sdk, name, key in ((anthropic, "Anthropic", "ANTHROPIC_API_KEY"), (groq, "Groq", "GROQ_API_KEY")):
        if isinstance(exc, sdk.AuthenticationError):
            return f"Invalid or missing {name} API key - check {key} in .env."
        if isinstance(exc, sdk.RateLimitError):
            return f"Rate limited by the {name} API - wait a moment and try again."
        if isinstance(exc, sdk.APIStatusError):
            return f"{name} API error {exc.status_code}: {exc.message}"
        if isinstance(exc, sdk.APIConnectionError):
            return f"Could not reach the {name} API - check your internet connection."
    return f"Agent error: {exc}"


def main() -> None:
    agent = MaskGuardAgent()
    print(f"Using {agent.label}")
    show = lambda name, args: print(f"  [tool] {name} {json.dumps(args)[:80]}")  # noqa: E731
    prompts = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    print("MaskGuard AI - ask about mask compliance (type 'exit' to quit).")
    while True:
        question = prompts.pop() if prompts else input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        try:
            print(f"\nMaskGuard: {agent.chat(question, on_tool=show)}")
        except Exception as exc:
            print(describe_api_error(exc))
        if prompts is not None and not prompts:
            break


if __name__ == "__main__":
    main()
