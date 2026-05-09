import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("rag")


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class TraceContext:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or generate_trace_id()
        self.stages: list[dict[str, Any]] = []
        self._start_time = time.time()

    @contextmanager
    def stage(self, name: str, **extra):
        stage_start = time.time()
        try:
            yield
        except Exception:
            latency_ms = int((time.time() - stage_start) * 1000)
            self.stages.append({
                "stage": name,
                "latency_ms": latency_ms,
                "error": True,
                **extra,
            })
            logger = get_logger()
            logger.error(
                f"{name} failed",
                extra={
                    "trace_id": self.trace_id,
                    "stage": name,
                    "latency_ms": latency_ms,
                    "extra_data": {"error": True, **extra},
                },
            )
            raise
        else:
            latency_ms = int((time.time() - stage_start) * 1000)
            stage_data = {
                "stage": name,
                "latency_ms": latency_ms,
                **extra,
            }
            self.stages.append(stage_data)
            logger = get_logger()
            logger.info(
                f"{name} completed",
                extra={
                    "trace_id": self.trace_id,
                    "stage": name,
                    "latency_ms": latency_ms,
                    "extra_data": extra,
                },
            )

    @property
    def total_latency_ms(self) -> int:
        return int((time.time() - self._start_time) * 1000)

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": self.total_latency_ms,
            "stages": self.stages,
        }
