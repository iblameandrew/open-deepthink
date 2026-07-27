__version__ = "0.1.12"
__release_name__ = "skill-code-entrypoints"
__release_tag__ = "0.1.12"

# deepthink package initialization
from .utils import clean_and_parse_json, execute_code_in_sandbox
from .state import GraphState, BRAINSTORM_EXPERTS

__all__ = [
    "__version__",
    "__release_name__",
    "__release_tag__",
    "clean_and_parse_json",
    "execute_code_in_sandbox",
    "GraphState",
    "BRAINSTORM_EXPERTS",
    "get_settings",
]


def __getattr__(name: str):
    """Lazy export for Settings to avoid import cycles at package load."""
    if name == "get_settings":
        from .config import get_settings

        return get_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
