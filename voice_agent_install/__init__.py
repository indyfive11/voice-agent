"""voice-agent-install — the thin-client provisioner (Phase-3 Layer B).

A flat, linear role-provisioner (NO plugin registry — that asymmetry lives only in
the brain's Layer C). Owns audio detection, LAN-brain discovery, and role emit
(``.env`` + systemd ``--user`` unit + linger) for the thin-client PoC.

Design of record: ``reference-installer-packaging-model`` (3-layer split; this is
Layer B). Discovery mechanism maintainer-locked 2026-07-13 (mDNS + manual floor behind an
ordered provider seam; UDP-broadcast rejected as correlated-failure). GA↔VAC
consensus 2026-07-20 (build 3a to the manual floor — zero brain dependency; write
the resolved LAN IP, never ``.local``).
"""
