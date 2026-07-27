"""Hardware detection. Pure stdlib.

The portability SOP governs this module:
  - Every value DEFAULTS to unset = works-exactly-as-today. A detector that can't tell returns 'none'/None,
    never a wrong guess (a wrong written value is worse than unset — cf. the 1.84× chipmunk-TTS regression,
    a per-startup probe that misfired). So this runs ONCE at setup and RETURNS values; it never writes
    config and it is never a per-startup probe.
  - Authoritative OS sources only: nvidia-smi / rocminfo, with an lspci fallback when the vendor tools
    aren't installed yet.

A.3 note: this is NET-NEW, not a generalization of gabagent's `config/setup.py:detect_gpu_env()`. That
legacy helper returns the AMD HSA-override string (or {} for BOTH nvidia and none) — it cannot express the
vendor. The STT-provider split (AMD → whisper.cpp/Vulkan, NVIDIA → CTranslate2/CUDA) needs the vendor
TRIPLE, so we report `amd | nvidia | none` here and carry the AMD override alongside it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["GpuInfo", "detect_gpu", "amd_hsa_override"]


@dataclass(frozen=True)
class GpuInfo:
    """The GPU facts the installer needs. `vendor` drives the STT split; `amd_override` is the suggested
    HSA_OVERRIDE_GFX_VERSION for a ROCm box (empty otherwise). `source` records how we decided, for the
    wizard to show its work."""
    vendor: str            # "amd" | "nvidia" | "none"
    amd_override: str = ""  # suggested HSA_OVERRIDE_GFX_VERSION, "" when N/A
    source: str = "none"    # "nvidia-smi" | "rocminfo" | "lspci" | "none"


def _run(argv: list[str], timeout: float = 10.0) -> str:
    """Best-effort capture of a probe's stdout. Never raises — a missing tool / timeout ⇒ ''."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def amd_hsa_override(rocminfo_out: str) -> str:
    """Derive the suggested HSA_OVERRIDE_GFX_VERSION 'X.Y.0' from rocminfo's gfx target, or '' if absent.

    'X.Y.Z' mirrors gfxXYZ with the trailing stepping folded to 0 (the family base ROCm ships):
    gfx1100/gfx1103 → 11.0.0, gfx1030 → 10.3.0, gfx900 → 9.0.0."""
    m = re.search(r"gfx(\d{3,4})\b", rocminfo_out)
    if not m:
        return ""
    g = m.group(1).zfill(4)
    return f"{int(g[:2])}.{int(g[2])}.0"


def detect_gpu() -> GpuInfo:
    """One-shot GPU vendor detection. Order: NVIDIA (nvidia-smi) → AMD (rocminfo) → lspci fallback → none.

    Returns a GpuInfo with vendor in {'amd','nvidia','none'}. Never raises; 'none' when undeterminable."""
    # 1. NVIDIA — nvidia-smi is authoritative and present iff a usable driver is installed.
    if shutil.which("nvidia-smi") and _run(["nvidia-smi", "-L"]).strip():
        return GpuInfo(vendor="nvidia", source="nvidia-smi")

    # 2. AMD — rocminfo is the authoritative ROCm source; also gives us the override suggestion.
    if shutil.which("rocminfo"):
        out = _run(["rocminfo"])
        if re.search(r"gfx\d{3,4}\b", out):
            return GpuInfo(vendor="amd", amd_override=amd_hsa_override(out), source="rocminfo")

    # 3. Fallback — no vendor tool installed yet. lspci can still name the vendor so the installer can
    #    provision the right STT path even on a fresh box. VGA/3D/Display controller lines only.
    lspci = _run(["lspci"])
    if lspci:
        for line in lspci.splitlines():
            if not re.search(r"\b(VGA|3D|Display)\b", line, re.I):
                continue
            if re.search(r"\bNVIDIA\b", line, re.I):
                return GpuInfo(vendor="nvidia", source="lspci")
            if re.search(r"\b(AMD|ATI|Radeon|Advanced Micro Devices)\b", line, re.I):
                return GpuInfo(vendor="amd", source="lspci")

    return GpuInfo(vendor="none", source="none")
