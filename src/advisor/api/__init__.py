"""FastAPI backend for the investment frontend.

Serves the React SPA plus a JSON API over the existing research engine and a
WebSocket that streams live TastyTrade equity quotes. Launched via `advisor web`.
"""

from advisor.api.app import create_app

__all__ = ["create_app"]
