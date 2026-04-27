"""Observability primitives tests."""
from __future__ import annotations

from fedloraguard.observability import get_logger, get_metrics


def test_metrics_renders_in_process_fallback():
    m = get_metrics()
    m.inc_counter("test_counter_total", verdict="benign")
    m.inc_counter("test_counter_total", verdict="malicious")
    m.observe_histogram("test_latency_ms", 12.3)
    rendered = m.render()
    assert "test_counter_total" in rendered
    assert "test_latency_ms" in rendered


def test_structured_logger_does_not_throw():
    log = get_logger("fedloraguard.test")
    log.info("event1", a=1, b="two")
    log.warn("event2", err="something")
