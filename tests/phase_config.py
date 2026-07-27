"""Phase config: typed Settings and defaults (Phase 1 production foundations)."""

from __future__ import annotations

import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-3:]
        results.append((name, "FAIL", f"{type(e).__name__}: {e} | " + " | ".join(tb)))


def t1():
    from deepthink.config import Settings, get_settings, reload_settings

    reload_settings()
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.default_provider in ("openrouter", "llamacpp")
    assert s.openrouter_model
    assert s.port == 8000 or isinstance(s.port, int)


chk("get_settings returns Settings", t1)


def t2():
    from deepthink.config import reload_settings

    s = reload_settings()
    qnn = s.qnn_defaults()
    assert "qnn_mode" in qnn and "manual_layers" in qnn
    qdad = s.qdad_defaults()
    assert qdad["grid_size"] >= 1 and "denoising_steps" in qdad
    prov = s.provider_defaults()
    assert prov["provider"] in ("openrouter", "llamacpp")
    assert prov["llamacpp_url"].endswith("/v1")


chk("settings default dicts for QNN/QDAD/provider", t2)


def t3():
    from deepthink.config import reload_settings

    s = reload_settings()
    url = s.normalize_llamacpp_url("http://localhost:8080")
    assert url.endswith("/v1")
    url2 = s.normalize_llamacpp_url("http://localhost:8080/v1/chat/completions")
    assert url2.endswith("/v1")
    assert "/chat/completions" not in url2


chk("normalize_llamacpp_url", t3)


def t4():
    from deepthink.config import reload_settings

    s = reload_settings()
    # Must not raise; may be None without env
    key = s.resolved_api_key()
    assert key is None or isinstance(key, str)


chk("resolved_api_key never crashes", t4)


def t5():
    assert (ROOT / "LICENSE").is_file()
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT" in text
    assert (ROOT / ".env.example").is_file()
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "docker-compose.yml").is_file()
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()


chk("LICENSE, .env.example, Docker, CI present", t5)


def t6():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "deepthink.cli:main" in pyproject or "deepthink.cli" in pyproject
    assert "pydantic-settings" in pyproject
    # Optional web extra should exist; core must not force FastAPI-only install
    assert "web" in pyproject or "[web]" in pyproject or "web =" in pyproject


chk("pyproject entry points and package list", t6)


def t7():
    from deepthink.config import Settings

    s = Settings(openrouter_api_key=None, api_key=None)
    assert s.resolved_api_key() is None
    s2 = Settings(openrouter_api_key="sk-test", api_key=None)
    assert s2.resolved_api_key() == "sk-test"


chk("Settings key resolution priority", t7)


for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE CONFIG: {ok}/{len(results)} OK")
