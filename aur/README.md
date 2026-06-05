# AUR packaging — `voice-agent-git`

Source-of-truth for the [AUR](https://aur.archlinux.org/) package. The AUR repo itself is a separate git
repo; these files are mirrored here for version control + review. Publish by copying `PKGBUILD`, `.SRCINFO`,
and `voice-agent-git.install` into the AUR clone and pushing.

## Why `-git`
No tagged releases yet, so this tracks `main`. (Once releases are tagged, a versioned `voice-agent` package can
source a release tarball instead.)

## Design
- **uv-bootstrapped.** The package ships only the source + a launcher (`arch=any`). On first run, the launcher
  `uv`-provisions the pinned Python (`>=3.12,<3.14` → 3.13, independent of the host's system Python) and the
  heavy ML deps (torch, onnxruntime, faster-whisper, kokoro, …) into a per-user venv. So the build is light and
  the package never fights the system Python.
- **Per-user working copy.** The app writes config/logs/model/venv in its working dir and isn't XDG-aware, so the
  launcher mirrors the pacman-managed `/usr/share/voice-agent` into `~/.local/share/voice-agent` (override with
  `VOICE_AGENT_HOME`) on first run and runs there. `voice-agent --update` re-syncs code after a package upgrade,
  preserving `.env`, `wakewords/`, `logs/`, `models/`, `data/`, and `.venv`.
- **Deps:** `uv` (interpreter + venv), `portaudio` (pyaudio), `espeak-ng` (Kokoro phonemizer), `rsync` (the
  copy/update). Optional: `pipewire` (echo-cancel source for wake-over-music), `ffmpeg` (wake training), `ollama`
  (local brain). `speexdsp-ns` (Speex NS) is NOT a dep — it's an off-by-default opt-in extra (cp312-only wheel).

## Test locally
```bash
cd aur
makepkg -si            # clone main, build, install (pulls depends via pacman)
# or, to just verify the build without installing deps:
makepkg -f --nodeps
namcap PKGBUILD *.pkg.tar.zst   # lint (pacman -S namcap)
```
Build artifacts (`src/`, `pkg/`, `*.pkg.tar.zst`, the clone dir) are gitignored — don't commit them.

## Publish / update on the AUR
```bash
# one-time: AUR account + SSH key registered at https://aur.archlinux.org/account
git clone ssh://aur@aur.archlinux.org/voice-agent-git.git aur-pub
cp PKGBUILD .SRCINFO voice-agent-git.install aur-pub/
cd aur-pub
makepkg --printsrcinfo > .SRCINFO     # regenerate so it matches PKGBUILD
git add PKGBUILD .SRCINFO voice-agent-git.install
git commit -m "voice-agent-git: <what changed>"
git push
```
`.SRCINFO` must always match `PKGBUILD` (the AUR rejects mismatches). `pkgver` auto-stamps on `makepkg` from
`git rev-list`/`rev-parse`; users get the latest `main` at install time regardless of the committed value.
