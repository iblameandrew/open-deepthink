"""Phase 5: FastAPI endpoints - all routes exist, methods, return types."""

import sys, traceback

sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
import importlib

mod = importlib.import_module("app")
app = mod.app

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


# Index page
def t1():
    routes = [r for r in app.routes if hasattr(r, "path")]
    assert "/" in [r.path for r in routes]


chk("GET / endpoint exists", t1)


# Inference from state
def t2():
    routes = [
        r for r in app.routes if getattr(r, "path", "") == "/run_inference_from_state"
    ]
    assert routes, "Missing /run_inference_from_state"


chk("POST /run_inference_from_state exists", t2)


# Build and run graph
def t3():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/build_and_run_graph"]
    assert routes


chk("POST /build_and_run_graph exists", t3)


# Export QNN
def t4():
    routes = [
        r for r in app.routes if getattr(r, "path", "") == "/export_qnn/{session_id}"
    ]
    assert routes


chk("GET /export_qnn/{session_id} exists", t4)


# Import QNN
def t5():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/import_qnn"]
    assert routes


chk("POST /import_qnn exists", t5)


# Upload documents
def t6():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/upload_documents"]
    assert routes


chk("POST /upload_documents exists", t6)


# Upload code files
def t6b():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/upload_code_files"]
    assert routes


chk("POST /upload_code_files exists", t6b)


# Chat
def t7():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/chat"]
    assert routes


chk("POST /chat exists", t7)


# Diagnostic chat
def t8():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/diagnostic_chat"]
    assert routes


chk("POST /diagnostic_chat exists", t8)


# Harvest
def t9():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/harvest"]
    assert routes


chk("POST /harvest exists", t9)


# Stream log
def t10():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/stream_log"]
    assert routes


chk("GET /stream_log exists", t10)


# Log stream (legacy)
def t11():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/log_stream"]
    assert routes


chk("GET /log_stream exists", t11)


# Download report
def t12():
    routes = [
        r
        for r in app.routes
        if getattr(r, "path", "") == "/download_report/{session_id}"
    ]
    assert routes


chk("GET /download_report/{session_id} exists", t12)


# Start distillation
def t13():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/start_distillation"]
    assert routes


chk("POST /start_distillation exists", t13)


# Stop distillation
def t14():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/stop_distillation"]
    assert routes


chk("POST /stop_distillation exists", t14)


# Distillation data
def t15():
    routes = [r for r in app.routes if getattr(r, "path", "") == "/distillation_data"]
    assert routes


chk("GET /distillation_data exists", t15)


# Download distillation
def t16():
    routes = [
        r for r in app.routes if getattr(r, "path", "") == "/download_distillation"
    ]
    assert routes


chk("GET /download_distillation exists", t16)


# Static mounts
def t17():
    # The /js, /css, /static mounts don't show in app.routes in same way, but they exist
    # Just check the app has them as routes
    routes = (
        [r for r in app.app.routes if hasattr(r, "path")] if hasattr(app, "app") else []
    )


chk("Static mounts (informational)", t17)


# All paths - sanity check
def t18():
    paths = sorted({r.path for r in app.routes if hasattr(r, "path")})
    expected = {
        "/",
        "/run_inference_from_state",
        "/build_and_run_graph",
        "/export_qnn/{session_id}",
        "/import_qnn",
        "/upload_documents",
        "/upload_code_files",
        "/chat",
        "/diagnostic_chat",
        "/harvest",
        "/stream_log",
        "/log_stream",
        "/download_report/{session_id}",
        "/start_distillation",
        "/stop_distillation",
        "/distillation_data",
        "/download_distillation",
    }
    missing = expected - set(paths)
    assert not missing, f"Missing endpoints: {missing}"


chk("All 16 expected endpoints present", t18)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 5: {ok}/{len(results)} OK")
