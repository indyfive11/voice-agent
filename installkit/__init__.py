"""installkit — the genuinely-shared installer core (Layer A).

Born at the Gab-Agent text-only workstation MVP (Phase-1). This package is the ONLY code shared
between the gabagent and voice-agent installers; it is later vendored into the voice-agent tree at a
pinned commit SHA (Phase-3), so it carries two hard invariants:

  1. IMPORT ISOLATION — installkit imports ONLY the Python standard library (plus anything it
     explicitly declares). It must NEVER import a consuming app (`gabagent`, `voice_agent`, …); a per-repo
     layer owns all app-specific config knowledge and writes. Enforced by tests/test_import_isolation.py.
  2. RETURN-DATA + THIN-WRITE — every engine RETURNS rendered data; a separate thin writer does the
     filesystem side, so a consumer's wizard can diff/preview before committing. No engine embeds an app
     name or an install-specific constant (portability SOP).

What lives here (Layer A, as of Phase-3 — the templating/token surface has now stabilized and the
vendor-pin is active):
  - wizard     — pure-stdlib interactive primitives (prompt / choose / confirm / save-confirm + panels).
  - deps       — the dependency-engine SYSTEM layer (distro detect, uv presence, system-pkg readiness).
  - hardware   — one-shot hardware detection that RETURNS values (never writes config); the GPU probe
                 reports the vendor triple `amd | nvidia | none` needed for the STT split.
  - templating — the artifact engine (A.4): render + atomic/mode-safe writers for `.env` / systemd
                 `--user` unit / `.desktop`. The unit renderer is the structural boot-safety chokepoint
                 (no Type=/Requires=/ExecStartPre → the Tier-0 boot-network violation is unexpressible).
  - secrets    — the token engine (A.5): ensure_tokens (authority: mint-when-absent) vs require_tokens
                 (consumer/satellite: fail-loud, never mint a shared secret the server never heard of).
"""

__all__ = ["wizard", "deps", "hardware", "templating", "secrets"]
__version__ = "0.1.0"
