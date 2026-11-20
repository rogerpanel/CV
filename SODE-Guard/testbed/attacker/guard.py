"""Public-target guard.

Every scenario resolves its target through here before sending a single packet.
If the target resolves to anything that is not a private (RFC1918 / loopback /
link-local) address, the run aborts. This is the last line of defence behind the
host iptables rule and the NetworkPolicy: it makes it impossible to point the
attacker tooling at an external host, whatever the config says.
"""
from __future__ import annotations

import ipaddress
import socket
import sys


def resolve(host: str) -> list[str]:
    host = host.split(":", 1)[0]
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise SystemExit(f"guard: cannot resolve {host!r}: {e}")
    return sorted({i[4][0] for i in infos})


def is_private(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    return ip.is_private or ip.is_loopback or ip.is_link_local


def assert_internal(host: str) -> str:
    """Return the first resolved address, or abort if any resolved address is public."""
    addrs = resolve(host)
    public = [a for a in addrs if not is_private(a)]
    if public:
        raise SystemExit(
            f"guard: REFUSING to attack {host!r}: resolves to public address(es) {public}. "
            "Testbed targets must be cluster-internal only."
        )
    if not addrs:
        raise SystemExit(f"guard: {host!r} resolved to nothing")
    return addrs[0]


if __name__ == "__main__":
    for h in sys.argv[1:]:
        print(f"{h} -> {assert_internal(h)} (internal OK)")
