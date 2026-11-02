#!/usr/bin/env python3
"""nanocode: a minimal DeepSeek coding agent built with LangChain only.

Requires: pip install -U langchain langchain-openai
Set DEEPSEEK_API_KEY before running.
"""

import glob as globlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

MODEL = os.getenv("MODEL", "deepseek-flash")
API_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, RED = "\033[34m", "\033[36m", "\033[32m", "\033[31m"


def confirm(action: str) -> bool:
    """Ask before changing files or executing shell commands."""
    try:
        answer = input(f"{RED}Allow {action}? [y/N] {RESET}").strip().lower()
        return answer in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


@tool("read")
def read_file(path: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Read a local text file and return numbered lines. offset is zero-based."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = max(0, offset)
    end = len(lines) if limit is None else start + max(0, limit)
    return "".join(f"{i + 1:4}| {lines[i]}" for i in range(start, min(end, len(lines)))) or "(empty)"


@tool("write")
def write_file(path: str, content: str) -> str:
    """Write text to a file, replacing any existing content. Requires user approval."""
    if not confirm(f"write to {path}"):
        return "denied by user"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "ok"


@tool("edit")
def edit_file(path: str, old: str, new: str, all: bool = False) -> str:
    """Replace exact text in a file. old must be unique unless all=True. Requires approval."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        return "error: old text not found"
    if count > 1 and not all:
        return f"error: old text occurs {count} times (set all=True to replace all)"
    if not confirm(f"edit {path}"):
        return "denied by user"
    updated = content.replace(old, new) if all else content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return "ok"


@tool("glob")
def find_files(pat: str, path: str = ".") -> str:
    """Find file paths matching a glob pattern, newest first. Use ** for recursion."""
    paths = globlib.glob(os.path.join(path, pat), recursive=True)
    files = [p for p in paths if os.path.isfile(p)]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return "\n".join(files[:100]) or "none"


@tool("grep")
def search_files(pat: str, path: str = ".") -> str:
    """Search text file lines using a Python regex; return file:line:content (up to 50)."""
    expression = re.compile(pat)
    hits = []
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ignored]
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                if os.path.getsize(filepath) > 2_000_000:
                    continue
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if expression.search(line):
                            hits.append(f"{filepath}:{lineno}:{line.rstrip()}")
                            if len(hits) >= 50:
                                return "\n".join(hits)
            except (OSError, UnicodeError):
                continue
    return "\n".join(hits) or "none"


@tool("bash")
def run_bash(cmd: str) -> str:
    """Execute a local shell command after user approval; timeout is 30 seconds."""
    if not confirm(f"shell command: {cmd}"):
        return "denied by user"
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            errors="replace", timeout=30,
        )
        out = (proc.stdout + proc.stderr).strip()
        return (out[:12000] or "(empty)") + f"\n(exit code: {proc.returncode})"
    except subprocess.TimeoutExpired as err:
        def decode(s):
            return s.decode("utf-8", errors="replace") if isinstance(s, bytes) else (s or "")
        return (decode(err.stdout) + decode(err.stderr))[:12000] + "\n(timed out after 30s)"


TOOLS = [read_file, write_file, edit_file, find_files, search_files, run_bash]


def separator() -> str:
    width = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)
    return f"{DIM}{'─' * width}{RESET}"


def print_new_messages(messages):
    for message in messages:
        kind = getattr(message, "type", "")
        if kind == "ai":
            for call in getattr(message, "tool_calls", []) or []:
                print(f"{GREEN}⏺ {call['name']}{RESET}({str(call['args'])[:120]})")
            if message.content:
                print(f"\n{CYAN}⏺{RESET} {message.content}")
        elif kind == "tool":
            content = str(message.content)
            preview = content.splitlines()[0][:100] if content else "(empty)"
            print(f"  {DIM}⎿ {preview}{' ...' if len(content.splitlines()) > 1 else ''}{RESET}")


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Please set DEEPSEEK_API_KEY")

    model = ChatOpenAI(
        model=MODEL,
        api_key=api_key,
        base_url=API_URL,
        max_tokens=8192,
        timeout=180,
        extra_body={"thinking": {"type": "disabled"}},
        use_responses_api=False,
    )
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=f"Concise coding assistant. Current working directory: {os.getcwd()}. "
                      "Use tools when useful. Do not assume command success without reading results.",
    )

    print(f"{BOLD}nanocode{RESET} | {DIM}LangChain + DeepSeek ({MODEL}) | {os.getcwd()}{RESET}")
    history = []
    while True:
        try:
            print(separator())
            question = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            if not question:
                continue
            if question in {"exit", "/q"}:
                break
            if question == "/c":
                history = []
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            old_count = len(history)
            result = agent.invoke(
                {"messages": history + [{"role": "user", "content": question}]},
                config={"recursion_limit": 50},
            )
            history = result["messages"]
            print_new_messages(history[old_count + 1:])
            print()
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"{RED}⏺ Error: {exc}{RESET}")


if __name__ == "__main__":
    main()
