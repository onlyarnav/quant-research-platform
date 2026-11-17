"""
Dashboard and API runtime configuration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure 'config' refers to top-level package and not this file if shadowed
if "config" in sys.modules and not hasattr(sys.modules["config"], "__path__"):
    del sys.modules["config"]

_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.settings import settings  # noqa: E402

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Runtime configuration for Dashboard API and Streamlit client."""

    api_url: str = settings.DASHBOARD_API_URL
    api_key: str = settings.DASHBOARD_API_KEY
    cache_ttl_seconds: int = settings.DASHBOARD_CACHE_TTL_SECONDS
    default_page_size: int = DEFAULT_PAGE_SIZE
    max_page_size: int = MAX_PAGE_SIZE
