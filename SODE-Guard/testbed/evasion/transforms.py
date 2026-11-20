"""Traffic transformations ("knobs"), each with a budget from the plan.

These realise an evasion as real traffic changes, not as edits to extracted
features. A knob vector is applied to the attacker pod before a scenario runs:

  | knob            | mechanism                                   | budget                |
  |-----------------|---------------------------------------------|-----------------------|
  | payload padding | extra junk HTTP header bytes per request    | <= +50% bytes         |
  | timing          | tc/netem delay + jitter on the pod egress   | <= +50 ms/pkt, <=3x T |
  | packet splitting| MSS clamp / small writes                    | <= 4x packet count    |
  | chaff           | benign-looking requests interleaved         | <= 1 chaff / attack   |
  | rate shaping    | attack rate (low-and-slow)                  | must stay functional  |

Timing and rate are enforced in the kernel with tc; padding, splitting and chaff
are passed to the HTTP-based scenarios as environment variables they honour. The
budgets are hard-capped here so a search can never exceed them.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

IFACE = "eth0"   # attacker pod egress


@dataclass
class Knobs:
    pad_frac: float = 0.0      # 0.0..0.5   extra bytes as fraction of request size
    delay_ms: float = 0.0      # 0..50      per-packet delay
    jitter_ms: float = 0.0     # 0..25      delay jitter
    mss: int = 0               # 0 = off; else clamp MSS (packet splitting)
    chaff_ratio: float = 0.0   # 0..1       chaff requests per attack request
    rate_mbit: float = 0.0     # 0 = off; else cap egress rate (low-and-slow)

    def clamp(self) -> "Knobs":
        self.pad_frac = min(max(self.pad_frac, 0.0), 0.5)
        self.delay_ms = min(max(self.delay_ms, 0.0), 50.0)
        self.jitter_ms = min(max(self.jitter_ms, 0.0), 25.0)
        self.mss = 0 if self.mss <= 0 else max(88, min(self.mss, 1460))
        self.chaff_ratio = min(max(self.chaff_ratio, 0.0), 1.0)
        self.rate_mbit = max(self.rate_mbit, 0.0)
        return self

    def as_env(self) -> dict[str, str]:
        return {"IDPS_PAD_FRAC": f"{self.pad_frac:.3f}",
                "IDPS_MSS": str(self.mss),
                "IDPS_CHAFF_RATIO": f"{self.chaff_ratio:.3f}"}

    def vector(self) -> list[float]:
        return [self.pad_frac, self.delay_ms, self.jitter_ms, float(self.mss),
                self.chaff_ratio, self.rate_mbit]

    @staticmethod
    def from_vector(v) -> "Knobs":
        return Knobs(v[0], v[1], v[2], int(v[3]), v[4], v[5]).clamp()


def _sh(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=False)


def apply_netem(k: Knobs) -> None:
    """Install tc/netem + tbf for timing, jitter and rate on the pod egress."""
    clear_netem()
    if k.delay_ms <= 0 and k.rate_mbit <= 0 and k.mss <= 0:
        return
    _sh(f"tc qdisc add dev {IFACE} root handle 1: htb default 10")
    rate = f"{k.rate_mbit:.3f}mbit" if k.rate_mbit > 0 else "1000mbit"
    _sh(f"tc class add dev {IFACE} parent 1: classid 1:10 htb rate {rate} ceil {rate}")
    if k.delay_ms > 0:
        jit = f" {k.jitter_ms:.1f}ms distribution normal" if k.jitter_ms > 0 else ""
        _sh(f"tc qdisc add dev {IFACE} parent 1:10 handle 10: netem delay {k.delay_ms:.1f}ms{jit}")
    if k.mss > 0:
        # packet splitting: clamp MSS so the stack emits more, smaller segments
        _sh(f"iptables -t mangle -A POSTROUTING -p tcp --tcp-flags SYN,RST SYN "
            f"-j TCPMSS --set-mss {k.mss}")


def clear_netem() -> None:
    _sh(f"tc qdisc del dev {IFACE} root 2>/dev/null")
    _sh("iptables -t mangle -F POSTROUTING 2>/dev/null")


def overhead(k: Knobs, base_bytes: int, base_pkts: int, base_dur: float) -> dict:
    """Measured cost of a knob setting, for the evasion-overhead column."""
    return {
        "extra_bytes_frac": k.pad_frac + k.chaff_ratio,
        "extra_pkts_frac": (base_pkts * (4 if k.mss else 1) - base_pkts) / max(base_pkts, 1),
        "extra_delay_ms": k.delay_ms,
        "rate_cap_mbit": k.rate_mbit,
    }
