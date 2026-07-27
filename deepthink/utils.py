"""
Core utilities and shared functions for DeepThink.
"""

import re
import json
import io
import asyncio
from contextlib import redirect_stdout, redirect_stderr


def clean_and_parse_json(llm_output_string):
    """
    Finds and parses the first valid JSON object within a string.
    Robustly handles:
    - Markdown code blocks
    - Trailing commas
    - C-style comments (// and /* */)
    - Unescaped newlines/tabs inside strings (common LLM error)
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output_string)
    if match:
        json_string = match.group(1)
    else:
        try:
            start_index = llm_output_string.index("{")
            end_index = llm_output_string.rindex("}") + 1
            json_string = llm_output_string[start_index:end_index]
        except ValueError:
            # print("Error: No JSON object found in the string.")
            return None

    # Step 1: Remove Comments (C-style) while preserving strings
    # Pattern captures: "string" OR //comment OR /*comment*/
    pattern = r'("(?:\\.|[^"\\])*")|//.*?$|/\*.*?\*/'

    def replace_comments(match):
        if match.group(1):  # It's a string, keep it
            return match.group(1)
        return ""  # It's a comment, remove it

    try:
        json_string = re.sub(pattern, replace_comments, json_string, flags=re.MULTILINE | re.DOTALL)
    except Exception:
        pass  # Fallback if regex fails (rare)

    # Step 2: Remove trailing commas before } or ]
    json_string = re.sub(r",\s*([}\]])", r"\1", json_string)

    # Step 3: Fix invalid escapes (e.g., \alpha, C:\Users).
    # Replaces a single \ NOT preceded by another \ (so we don't touch already
    # valid \\ pairs) and NOT followed by a valid JSON escape char. This prevents
    # "Invalid \escape" errors caused by LLM-generated Windows paths like
    # C:\Users\foo without breaking already-escaped backslash pairs.
    try:
        # The (?:\\ ) non-capturing group is required; the bare `\\` form after
        # the lookbehind is mis-parsed by the `re` module and matches the first
        # backslash of an already-escaped pair.
        json_string = re.sub(r'(?<!\\)(?:\\)(?![\\"/bfnrtu])', r"\\\\", json_string)
    except Exception:
        pass

    # Step 4: Attempt fast load
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    # Step 5: Fix Unescaped Control Characters in Strings (Fallback)
    # This manually iterates to find strings and replace literal \n with \\n
    new_chars = []
    in_string = False
    escaped = False
    for char in json_string:
        if char == '"' and not escaped:
            in_string = not in_string
            new_chars.append(char)
            escaped = False
        elif in_string:
            if char == "\n":
                new_chars.append("\\n")
            elif char == "\t":
                new_chars.append("\\t")
            elif char == "\r":
                pass  # Skip CR
            elif char == "\\":
                escaped = not escaped
                new_chars.append(char)
            else:
                escaped = False
                new_chars.append(char)
        else:
            new_chars.append(char)
            escaped = False

    repaired_string = "".join(new_chars)

    try:
        return json.loads(repaired_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON even after repair: {e}")
        # print(f"Problematic string start: {json_string[:200]}")
        return None


def execute_code_in_sandbox(code: str) -> tuple:
    """
    Executes a string of Python code and captures its stdout/stderr.
    Returns a tuple of (success: bool, output: str).
    """
    if not code:
        return True, "No code to execute."

    # Extract code from markdown block if present
    code_match = re.search(r"```(?:python\n)?([\s\S]*?)```", code)
    if code_match:
        code = code_match.group(1).strip()

    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # Using a restricted globals dict for a little more safety
            exec(
                code,
                {
                    "__builtins__": {
                        "print": print,
                        "range": range,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "tuple": tuple,
                        "True": True,
                        "False": False,
                        "None": None,
                    }
                },
            )
        return True, output_buffer.getvalue()
    except Exception as e:
        return False, f"{output_buffer.getvalue()}\n\nERROR: {type(e).__name__}: {e}"
