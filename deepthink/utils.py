"""
Core utilities and shared functions for DeepThink.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def clean_and_parse_json(llm_output_string):
    """
    Finds and parses the first valid JSON object within a string.
    Robustly handles:
    - Markdown code blocks
    - Trailing commas
    - C-style comments (// and /* */)
    - Unescaped newlines/tabs inside strings (common LLM error)
    """
    if llm_output_string is None:
        return None
    if isinstance(llm_output_string, dict):
        return llm_output_string
    if not isinstance(llm_output_string, str):
        llm_output_string = str(llm_output_string)

    match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output_string)
    if match:
        json_string = match.group(1)
    else:
        try:
            start_index = llm_output_string.index("{")
            end_index = llm_output_string.rindex("}") + 1
            json_string = llm_output_string[start_index:end_index]
        except ValueError:
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
    except json.JSONDecodeError:
        return None


def parse_llm_json(raw: Any, default: dict | None = None) -> dict:
    """Parse an LLM payload into a dict (fenced / messy JSON included)."""
    if isinstance(raw, dict):
        return raw
    parsed = clean_and_parse_json(raw)
    if isinstance(parsed, dict):
        return parsed
    return {} if default is None else default


_TIKTOKEN_ENC = None
_TIKTOKEN_FAILED = False


def estimate_tokens(text: Any) -> int:
    """
    Count tokens for budget accounting.

    Prefers tiktoken ``cl100k_base`` (OpenRouter / OpenAI-compatible). Falls
    back to a ~4-chars-per-token heuristic when tiktoken is unavailable.
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0

    global _TIKTOKEN_ENC, _TIKTOKEN_FAILED
    if not _TIKTOKEN_FAILED:
        try:
            if _TIKTOKEN_ENC is None:
                import tiktoken

                _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
            return len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))
        except Exception:
            _TIKTOKEN_FAILED = True
    return max(1, (len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# Isolated code sandbox
# ---------------------------------------------------------------------------


class SandboxError(Exception):
    """Raised when user code is rejected before execution."""


_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "memoryview",
        "help",
        "exit",
        "quit",
        "copyright",
        "credits",
        "license",
        "__builtins__",
        "__loader__",
        "__spec__",
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "importlib",
        "ctypes",
        "multiprocessing",
        "builtins",
    }
)

_SAFE_BUILTINS = (
    "print",
    "range",
    "len",
    "str",
    "int",
    "float",
    "list",
    "dict",
    "set",
    "tuple",
    "bool",
    "abs",
    "min",
    "max",
    "sum",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "round",
    "isinstance",
    "any",
    "all",
    "ord",
    "chr",
    "bin",
    "hex",
    "oct",
    "pow",
    "divmod",
    "repr",
    "format",
    "slice",
    "iter",
    "next",
    "True",
    "False",
    "None",
)


class _SandboxVisitor(ast.NodeVisitor):
    def visit_Import(self, node):
        raise SandboxError("imports are not allowed")

    def visit_ImportFrom(self, node):
        raise SandboxError("imports are not allowed")

    def visit_Attribute(self, node):
        if isinstance(node.attr, str) and node.attr.startswith("_"):
            raise SandboxError(f"attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
            raise SandboxError(f"name {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node):
        raise SandboxError("global is not allowed")

    def visit_Nonlocal(self, node):
        raise SandboxError("nonlocal is not allowed")


def _assert_sandbox_ast_safe(code: str) -> ast.AST:
    try:
        tree = ast.parse(code, filename="<sandbox>", mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"SyntaxError: {e}") from e
    _SandboxVisitor().visit(tree)
    return tree


def _sandbox_env() -> dict:
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]
    else:
        env["PATH"] = "/usr/bin:/bin"
        env["HOME"] = tempfile.gettempdir()
        env["TMPDIR"] = tempfile.gettempdir()
    return env


_SANDBOX_RUNNER = r"""
import ast
import io
import sys
from contextlib import redirect_stderr, redirect_stdout

FORBIDDEN = frozenset({
    "__import__", "open", "exec", "eval", "compile", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "hasattr",
    "type", "object", "super", "classmethod", "staticmethod", "property",
    "memoryview", "help", "exit", "quit", "copyright", "credits", "license",
    "__builtins__", "__loader__", "__spec__", "os", "sys", "subprocess",
    "socket", "shutil", "pathlib", "importlib", "ctypes", "multiprocessing",
    "builtins",
})

class V(ast.NodeVisitor):
    def visit_Import(self, node):
        raise RuntimeError("imports are not allowed")
    def visit_ImportFrom(self, node):
        raise RuntimeError("imports are not allowed")
    def visit_Attribute(self, node):
        if isinstance(node.attr, str) and node.attr.startswith("_"):
            raise RuntimeError("attribute %r is not allowed" % (node.attr,))
        self.generic_visit(node)
    def visit_Name(self, node):
        if node.id in FORBIDDEN or node.id.startswith("__"):
            raise RuntimeError("name %r is not allowed" % (node.id,))
        self.generic_visit(node)
    def visit_Global(self, node):
        raise RuntimeError("global is not allowed")
    def visit_Nonlocal(self, node):
        raise RuntimeError("nonlocal is not allowed")

SAFE = {
    "print": print, "range": range, "len": len, "str": str, "int": int,
    "float": float, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "bool": bool, "abs": abs, "min": min, "max": max, "sum": sum,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "round": round,
    "isinstance": isinstance, "any": any, "all": all, "ord": ord, "chr": chr,
    "bin": bin, "hex": hex, "oct": oct, "pow": pow, "divmod": divmod,
    "repr": repr, "format": format, "slice": slice, "iter": iter, "next": next,
    "True": True, "False": False, "None": None,
}

src_path = sys.argv[1]
with open(src_path, "r", encoding="utf-8") as fh:
    code = fh.read()
try:
    tree = ast.parse(code, filename="<sandbox>", mode="exec")
    V().visit(tree)
except Exception as e:
    sys.stderr.write("%s: %s\n" % (type(e).__name__, e))
    sys.exit(2)

buf = io.StringIO()
try:
    with redirect_stdout(buf), redirect_stderr(buf):
        exec(compile(tree, "<sandbox>", "exec"), {"__builtins__": SAFE}, {})
except Exception as e:
    sys.stdout.write(buf.getvalue())
    sys.stderr.write("ERROR: %s: %s\n" % (type(e).__name__, e))
    sys.exit(1)
sys.stdout.write(buf.getvalue())
"""


def _extract_sandbox_source(code: str) -> str:
    code_match = re.search(r"```(?:python\n)?([\s\S]*?)```", code)
    if code_match:
        return code_match.group(1).strip()
    return code.strip()


def execute_code_in_sandbox(code: str, timeout: float = 5.0) -> tuple:
    """
    Execute Python in an isolated subprocess after AST rejection of imports
    and dunder / filesystem escapes.

    Returns ``(success: bool, output: str)``.
    """
    if not code:
        return True, "No code to execute."

    source = _extract_sandbox_source(code)
    if not source:
        return True, "No code to execute."

    try:
        _assert_sandbox_ast_safe(source)
    except SandboxError as e:
        return False, f"ERROR: {type(e).__name__}: {e}"

    try:
        with tempfile.TemporaryDirectory(prefix="odt-sandbox-") as tmp:
            tmp_path = Path(tmp)
            user_path = tmp_path / "user.py"
            runner_path = tmp_path / "_runner.py"
            user_path.write_text(source, encoding="utf-8")
            runner_path.write_text(_SANDBOX_RUNNER, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", str(runner_path), str(user_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmp_path),
                env=_sandbox_env(),
            )
    except subprocess.TimeoutExpired:
        return False, f"ERROR: TimeoutError: sandbox exceeded {timeout}s"
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}"

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, output or f"ERROR: sandbox exited {proc.returncode}"
    return True, output
