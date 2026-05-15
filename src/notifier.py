"""
notifier.py -- Optional Telegram alert bot.

Reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from the
environment.  If either is missing the notifier becomes a no-op
(``enabled = False``).  Network calls run in a background thread so
the async scanner is never blocked.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - requests is in requirements.txt
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Fire-and-forget Telegram notifier."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id and requests is not None)
        if not self.enabled:
            logger.info("TelegramNotifier disabled (missing token/chat_id or requests)")

    # ───────────────────────────────────────────
    #  Low-level send
    # ───────────────────────────────────────────

    def _post(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            url = _TELEGRAM_API.format(token=self.token)
            resp = requests.post(  # type: ignore[union-attr]
                url,
                json={"chat_id": self.chat_id, "text": text},
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Telegram send failed: %s %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:  # pragma: no cover - network
            logger.warning("Telegram send error: %s", exc)

    def send(self, text: str) -> None:
        """Fire the request from a daemon thread; never blocks."""
        if not self.enabled:
            return
        threading.Thread(
            target=self._post,
            args=(text,),
            daemon=True,
        ).start()

    # ───────────────────────────────────────────
    #  High-level helpers
    # ───────────────────────────────────────────

    @staticmethod
    def format_ev_alert(row: dict[str, Any]) -> str:
        return (
            f"+EV Alert: {row.get('Outcome', '?')} "
            f"{row.get('Offered_Odds', '?')} @ {row.get('Bookmaker', '?')} | "
            f"EV: {row.get('EV_%', '?')}% | "
            f"Kelly: {row.get('Kelly_Bet', '?')}"
        )

    @staticmethod
    def format_arb_alert(row: dict[str, Any]) -> str:
        return (
            f"ARB: {row.get('Matchup', '?')} | "
            f"Margin: {row.get('Margin_%', '?')}% | "
            f"Stakes: {row.get('Stake_1', '?')} / {row.get('Stake_2', '?')}"
        )

    @staticmethod
    def format_steam_alert(signal: Any) -> str:
        # Accepts a SteamSignal dataclass or any object with .message().
        msg = getattr(signal, "message", None)
        return msg() if callable(msg) else str(signal)
