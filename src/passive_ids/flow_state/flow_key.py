"""5-tuple flow key, shared by provisional and confirmed tables.

A plain frozen dataclass rather than a plain tuple so field names are
self-documenting at every call site -- `key.dst_port`, not `key[3]`.
Hashable and immutable, which is what both tables need to use it
directly as a dict key.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def canonical(self) -> tuple["FlowKey", bool]:
        """Normalizes both directions of one connection onto the same
        key. Without this, a client->server packet and the server's
        response carry different (src, dst)-swapped FlowKeys and land
        in two unrelated table entries -- verified directly: a 3+3
        packet exchange produced 2 separate ConfirmedFlowEntry objects
        instead of 1, which silently breaks anything that needs to
        compare outbound vs inbound traffic for the same connection
        (exfiltration detection's entire signal) and can delay or
        prevent promotion for flows that split evenly between
        directions.

        Returns (canonical_key, is_forward). canonical_key is
        identical for both directions of a connection -- whichever
        (ip, port) pair sorts first becomes its src. is_forward is
        True if this key's own orientation already matches that
        canonical form, False if it's the reverse direction -- callers
        use this to attribute byte/packet counts to the correct side.
        """
        this_end = (self.src_ip, self.src_port)
        other_end = (self.dst_ip, self.dst_port)
        if this_end <= other_end:
            return self, True
        reversed_key = FlowKey(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            protocol=self.protocol,
        )
        return reversed_key, False
