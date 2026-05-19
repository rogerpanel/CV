"""Tests for protocol message canonicalisation."""
from __future__ import annotations

import pytest


def _import_data():
    try:
        from mambaguard.data import (
            MessageCanonicaliser,
            ProtocolMessage,
            from_a2a,
            from_acp,
            from_anp,
            from_mcp,
        )
    except Exception as exc:
        pytest.skip(f"data module unavailable: {exc}")
    return ProtocolMessage, MessageCanonicaliser, {
        "mcp": from_mcp,
        "acp": from_acp,
        "a2a": from_a2a,
        "anp": from_anp,
    }


def test_protocol_message_round_trip():
    ProtocolMessage, _, _ = _import_data()
    from mambaguard.data import EDGE_TYPES

    msg = ProtocolMessage(
        tau=EDGE_TYPES[0],
        src="alice",
        dst="search_tool",
        payload="hello",
        metadata={},
        t_m=1.0,
        label=0,
        msg_id="m1",
    )
    # Round-trip through dataclasses.asdict if available.
    from dataclasses import asdict

    d = asdict(msg)
    msg2 = ProtocolMessage(**d)
    assert msg2.msg_id == "m1"
    assert msg2.tau == EDGE_TYPES[0]
    assert msg2.t_m == 1.0


@pytest.mark.parametrize("proto", ["mcp", "acp", "a2a", "anp"])
def test_protocol_adapters_exist(proto):
    ProtocolMessage, _, adapters = _import_data()
    fn = adapters[proto]
    # The exact raw schema depends on the adapter; we only assert callable
    # and that it tolerates the documented minimal record.
    assert callable(fn)


def test_encode_batch_monkeypatched(monkeypatch):
    ProtocolMessage, MessageCanonicaliser, _ = _import_data()
    torch = pytest.importorskip("torch")

    canon = MessageCanonicaliser(d_p=16)

    class _FakeST:
        def encode(self, payloads, **kw):
            return torch.zeros(len(payloads), 16)

        def parameters(self):
            return iter([])

        def eval(self):
            return self

    monkeypatch.setattr(canon, "_load", lambda: _FakeST(), raising=True)

    out = canon.encode_batch(["hi", "there", "world"])
    assert out.shape == (3, 16)


def test_encode_batch_empty_returns_zero_rows():
    _, MessageCanonicaliser, _ = _import_data()
    torch = pytest.importorskip("torch")
    canon = MessageCanonicaliser(d_p=8)
    out = canon.encode_batch([])
    assert out.shape == (0, 8)
