import sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from deepthink.utils import (
    clean_and_parse_json,
    estimate_tokens,
    execute_code_in_sandbox,
    parse_llm_json,
)
results = []
def chk(name, fn):
    try:
        fn(); results.append((name, "OK", None))
    except AssertionError as e: results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e: results.append((name, "FAIL", f"{type(e).__name__}: {e}"))

def t1(): assert clean_and_parse_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
chk("basic JSON", t1)
def t2():
    res = clean_and_parse_json('Here is the JSON: ```json\n{"x": "y"}\n``` done')
    assert res == {"x": "y"}, f"got {res}"
chk("markdown code block", t2)
def t3():
    res = clean_and_parse_json('result: {"a": 1, "b": 2,}, tail')
    assert res == {"a": 1, "b": 2}, f"got {res}"
chk("trailing commas", t3)
def t4():
    s = 'noise {"k": "value with // slash and /* comment */", "ok": true} more'
    res = clean_and_parse_json(s)
    assert res is not None and "k" in res, f"got {res}"
chk("C-style comments in string preserved", t4)
def t5(): assert clean_and_parse_json('prefix {"a": "b"} suffix') == {"a": "b"}
chk("embedded JSON without code fence", t5)
def t6():
    assert clean_and_parse_json("") is None
    assert clean_and_parse_json("no json here") is None
chk("no JSON returns None", t6)
def t7():
    s = '{"a": "line1\nline2"}'
    res = clean_and_parse_json(s)
    assert res == {"a": "line1\nline2"}, f"got {res}"
chk("unescaped newline in string repaired", t7)
def t8():
    s = '{"path": "C:\\\\Users\\\\foo"}'
    res = clean_and_parse_json(s)
    assert res is not None, f"got {res}"
chk("Windows-style path with backslashes", t8)

def t9():
    success, out = execute_code_in_sandbox("print('hello')")
    assert success and "hello" in out
chk("simple print", t9)
def t10():
    success, out = execute_code_in_sandbox("```python\nprint('hi')\n```")
    assert success and "hi" in out
chk("markdown code fence extracted", t10)
def t11():
    success, out = execute_code_in_sandbox("raise ValueError('boom')")
    assert not success and "ValueError" in out
chk("exception captured", t11)
def t12():
    success, out = execute_code_in_sandbox("")
    assert success and "No code" in out
chk("empty code returns no-op success", t12)
def t13():
    success, out = execute_code_in_sandbox("x = 1 + 1\nprint(x)")
    assert success and "2" in out
chk("simple arithmetic", t13)
def t14():
    success, _ = execute_code_in_sandbox("import os\nprint(os.getcwd())")
    assert not success, "Sandbox must NOT allow unrestricted imports"
chk("sandbox blocks `import os` (security)", t14)
def t15():
    success, _ = execute_code_in_sandbox("open('test.txt')")
    assert not success, "Sandbox must NOT allow open()"
chk("sandbox blocks open() (security)", t15)

def t16():
    success, _ = execute_code_in_sandbox("print(('').__class__.__mro__)")
    assert not success, "Sandbox must NOT allow dunder attribute walks"
chk("sandbox blocks dunder attribute escape", t16)

def t18():
    n = estimate_tokens("hello world")
    assert isinstance(n, int) and n >= 1
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    long_n = estimate_tokens("word " * 200)
    assert long_n > n
chk("estimate_tokens returns positive counts", t18)

def t19():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json({"already": True}) == {"already": True}
    assert parse_llm_json("not json") == {}
    assert parse_llm_json("not json", default={"ok": False}) == {"ok": False}
chk("parse_llm_json handles messy LLM payloads", t19)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err: line += f" :: {err}"
    print(line)
ok = sum(1 for _,s,_ in results if s == "OK")
print(f"\nPHASE 2: {ok}/{len(results)} OK")
