#!/usr/bin/env python3
"""nanocode - minimal coding agent powered by the DeepSeek API."""

import glob as globlib
import json
import os
import re
import shutil
import subprocess

from openai import OpenAI

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com"
MODEL = os.environ.get("MODEL", "deepseek-flash")

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, RED = "\033[34m", "\033[36m", "\033[32m", "\033[31m"


def read(args):
    with open(args["path"], encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    offset = max(0, args.get("offset", 0))
    limit = max(0, args.get("limit", len(lines)))
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(lines[offset:offset + limit]))


def write(args):
    with open(args["path"], "w", encoding="utf-8") as f:
        f.write(args["content"])
    return "ok"


def edit(args):
    with open(args["path"], encoding="utf-8") as f:
        content = f.read()
    old, new = args["old"], args["new"]
    if not old:
        return "error: old_string cannot be empty"
    if old not in content:
        return "error: old_string not found"
    count = content.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    updated = content.replace(old, new) if args.get("all") else content.replace(old, new, 1)
    with open(args["path"], "w", encoding="utf-8") as f:
        f.write(updated)
    return "ok"


def glob(args):
    pattern = os.path.join(args.get("path", "."), args["pat"])
    files = globlib.glob(pattern, recursive=True)
    files.sort(key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0, reverse=True)
    return "\n".join(files) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    root = args.get("path", ".")
    for filepath in globlib.iglob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
                        if len(hits) >= 50:
                            return "\n".join(hits)
        except (OSError, UnicodeError):
            pass
    return "\n".join(hits) or "none"


def bash(args):
    try:
        proc = subprocess.run(
            args["cmd"], shell=True, capture_output=True, text=True,
            errors="replace", timeout=30,
        )
        output = (proc.stdout + proc.stderr).strip()
        for line in output.splitlines():
            print(f"  {DIM}│ {line}{RESET}", flush=True)
        return (output or "(empty)") + f"\n(exit code: {proc.returncode})"
    except subprocess.TimeoutExpired as err:
        captured = err.stdout or b""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        return captured + "\n(timed out after 30s)"

def calculate(args):
    a = args["input"]
    return str(100 - a)

def haha(args):
    a = args["input"]
    return str(200 - a)


TOOLS = {
    "read": ("Read file with line numbers (file path, not directory)", {"path": "string", "offset": "number?", "limit": "number?"}, read),
    "write": ("Write content to file", {"path": "string", "content": "string"}, write),
    "edit": ("Replace old with new in file (old must be unique unless all=true)", {"path": "string", "old": "string", "new": "string", "all": "boolean?"}, edit),
    "glob": ("Find files by pattern, sorted by mtime", {"pat": "string", "path": "string?"}, glob),
    "grep": ("Search files for regex pattern", {"pat": "string", "path": "string?"}, grep),
    "bash": ("Run shell command", {"cmd": "string"}, bash),
    "calculate": ("计算任意一个数的幸运数", {"input": "number"}, calculate),
    "haha": ("计算任意一个数的悲观数", {"input": "number"}, haha)
}


def run_tool(name, args):
    if name not in TOOLS:
        return f"error: unknown tool {name}"
    if not isinstance(args, dict):
        return "error: tool arguments must be a JSON object"
    try:
        return str(TOOLS[name][2](args))
    except Exception as err:
        return f"error: {err}"


def make_schema():
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties, required = {}, []
        for param_name, param_type in params.items():
            optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {"type": "integer" if base_type == "number" else base_type}
            if not optional:
                required.append(param_name)
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        })
    return result


def send_messages(client, messages):
    """Send conversation to DeepSeek and return the SDK ChatCompletionMessage."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=make_schema(),
        tool_choice="auto",
        max_tokens=8192,
        extra_body={"thinking": {"type": "disabled"}},
    )
    if not response.choices:
        raise RuntimeError("DeepSeek API returned no choices")
    return response.choices[0].message


def separator():
    width = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)
    return f"{DIM}{'─' * width}{RESET}"


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", text, flags=re.DOTALL)


def main():
    print(f"{BOLD}nanocode{RESET} | {DIM}{MODEL} (DeepSeek) | {os.getcwd()}{RESET}\n")
    if not DEEPSEEK_KEY:
        print(f"{RED}Please set DEEPSEEK_API_KEY before running.{RESET}")
        return
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url=API_URL, timeout=180.0)
    system_prompt = (
        f"Concise coding assistant. cwd: {os.getcwd()}. "
        "You may use tools to inspect, modify, and test code. "
        "Use the available tools when needed and summarize the outcome."
    )

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            if not user_input:
                continue
            if user_input in ("/q", "exit"):
                break
            if user_input == "/c":
                messages = [{"role": "system", "content": system_prompt}]
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            messages.append({"role": "user", "content": user_input})
            while True:
                # Same OpenAI SDK tool-calling workflow as the official example.
                message = send_messages(client, messages)
                if message.content:
                    print(f"\n{CYAN}⏺{RESET} {render_markdown(message.content)}")

                # Save the assistant message, including any tool_calls, verbatim.
                # SDK objects must be converted to dictionaries for the next request.
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break

                # One response can request multiple tools; execute every call.
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
                        preview_keys = {
                            "read": "path", "write": "path", "edit": "path",
                            "glob": "pat", "grep": "pat", "bash": "cmd",
                        }
                        key = preview_keys.get(tool_name)
                        preview_value = tool_args.get(key, "") if isinstance(tool_args, dict) else ""
                        arg_preview = str(preview_value)[:50]
                        print(f"\n{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})")
                        result = run_tool(tool_name, tool_args)
                    except (ValueError, TypeError) as err:
                        result = f"error: invalid tool arguments: {err}"
                        print(f"\n{RED}⏺ Invalid tool call: {err}{RESET}")

                    result_lines = result.splitlines() or [""]
                    preview = result_lines[0][:60]
                    if len(result_lines) > 1:
                        preview += f" ... +{len(result_lines) - 1} lines"
                    elif len(result_lines[0]) > 60:
                        preview += "..."
                    print(f"  {DIM}⎿  {preview}{RESET}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            print()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")
            # Prevent replaying an unfinished tool call sequence after an API error.
            messages = [{"role": "system", "content": system_prompt}]


if __name__ == "__main__":
    main()
