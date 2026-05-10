from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any


_DEBUG_TRUE_VALUES = {"1", "true", "yes", "y", "on", "debug"}
_CONFIGURED = False


class UtcIsoFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", self.converter(record.created))




class StageFieldFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = "app"
        return True

class StageLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.setdefault("stage", self.extra["stage"])
        kwargs["extra"] = extra
        return msg, kwargs


def _env_debug_enabled() -> bool:
    return str(os.getenv("ACCOUNTING_DEBUG", "0")).strip().lower() in _DEBUG_TRUE_VALUES


def resolve_log_level(default: str = "INFO") -> int:
    if _env_debug_enabled():
        return logging.DEBUG

    raw = str(os.getenv("ACCOUNTING_LOG_LEVEL", default)).strip().upper() or default
    return getattr(logging, raw, logging.INFO)


def configure_logging(*, default_level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(StageFieldFilter())
    handler.setFormatter(UtcIsoFormatter("%(asctime)s %(levelname)s [%(stage)s] %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolve_log_level(default_level))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(stage: str) -> StageLoggerAdapter:
    configure_logging()
    return StageLoggerAdapter(logging.getLogger(stage), {"stage": stage})
