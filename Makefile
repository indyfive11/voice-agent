# Vendored Layer-A shared installer core (installkit).
#
# installkit is DEVELOPED in its own repo and VENDORED into this tree at a pinned commit, per the locked
# packaging model (INSTALL_PLAN §4/§9). The app imports `installkit` from the vendored copy at the repo
# root (installkit/); nothing depends on the sibling checkout at runtime, and the Pi rsync ships the
# vendored copy because it is git-tracked (`git ls-files '*.py'`).
#
# PUBLIC PIN: installkit is published, so INSTALLKIT_PIN 78ff1cd is a REAL PUBLIC commit on
# github.com/indyfive11/installkit. `vendor-check` content-derives the SHA-match by comparing the vendored
# bytes against installkit@INSTALLKIT_PIN in INSTALLKIT_SRC — a local checkout OR a fresh CI clone that
# contains the pinned commit. Wired as a blocking CI gate in .github/workflows/vendor-check.yml.

# bash for the tree-OID gate's `<(...)` process substitution (comm on the two file listings).
SHELL := /bin/bash

INSTALLKIT_PIN ?= 78ff1cd
INSTALLKIT_SRC ?= $(HOME)/dev/installkit
VENDOR_DIR     := installkit
MODULES        := __init__.py deps.py hardware.py secrets.py templating.py wizard.py

.PHONY: vendor-sync vendor-check installer-parity aur-parity bootstrap-smoke install-hooks

vendor-sync:   ## Re-copy installkit modules from $(INSTALLKIT_SRC) at $(INSTALLKIT_PIN) into $(VENDOR_DIR)/
	@for m in $(MODULES); do \
	  git -C "$(INSTALLKIT_SRC)" show "$(INSTALLKIT_PIN):installkit/$$m" > "$(VENDOR_DIR)/$$m" \
	    && echo "vendored $$m @ $(INSTALLKIT_PIN)"; \
	done
	@echo "Done. Review 'git diff $(VENDOR_DIR)/' and commit (maintainer-gated)."

vendor-check:  ## Fail if vendored installkit/ diverges from installkit@$(INSTALLKIT_PIN) (content + file-set + modes)
	@rc=0; \
	want=$$(git -C "$(INSTALLKIT_SRC)" rev-parse "$(INSTALLKIT_PIN):installkit" 2>/dev/null); \
	have=$$(git rev-parse "HEAD:$(VENDOR_DIR)" 2>/dev/null); \
	if [ -z "$$want" ]; then \
	  echo "ERROR cannot resolve installkit@$(INSTALLKIT_PIN):installkit in $(INSTALLKIT_SRC)"; rc=1; \
	elif [ "$$want" = "$$have" ]; then \
	  echo "OK    tree  $(VENDOR_DIR)/ == installkit@$(INSTALLKIT_PIN):installkit ($$want)"; \
	else \
	  echo "DRIFT tree  $(VENDOR_DIR)/ ($$have) != installkit@$(INSTALLKIT_PIN):installkit ($$want)"; rc=1; \
	  wl=$$(git -C "$(INSTALLKIT_SRC)" ls-tree --name-only "$(INSTALLKIT_PIN):installkit" | sort); \
	  hl=$$(git ls-tree --name-only "HEAD:$(VENDOR_DIR)" | sort); \
	  comm -23 <(printf '%s\n' "$$wl") <(printf '%s\n' "$$hl") | sed 's/^/      MISSING (in pin, not vendored): /'; \
	  comm -13 <(printf '%s\n' "$$wl") <(printf '%s\n' "$$hl") | sed 's/^/      EXTRA   (vendored, not in pin): /'; \
	fi; \
	for m in $(MODULES); do \
	  if git -C "$(INSTALLKIT_SRC)" show "$(INSTALLKIT_PIN):installkit/$$m" 2>/dev/null | cmp -s - "$(VENDOR_DIR)/$$m"; then \
	    echo "OK    $$m"; \
	  else \
	    echo "DRIFT $$m  (vendored copy != installkit@$(INSTALLKIT_PIN))"; rc=1; \
	  fi; \
	done; \
	[ $$rc -eq 0 ] || echo "vendor-check FAILED — run 'make vendor-sync' or reconcile the pin."; \
	exit $$rc

# --- Installer-parity gates (SOP: CLAUDE.md "Installer parity") ---

installer-parity:  ## Run the stateless parity pytest + the config-knob (canary/ignore-lint) gate
	uv run python -m pytest -q tests/test_installer_parity.py
	bash scripts/installer-parity.sh

aur-parity:  ## Cross-repo AUR bridge. `make aur-parity` = skip-loud if clone absent; BUMP=1 = hard-fail.
	python3 tools/aur_parity.py $(if $(BUMP),--bump,)

bootstrap-smoke:  ## Reachability backstop: only the git-tracked tree must reach every install surface
	bash tools/bootstrap_smoke.sh

install-hooks:  ## Install the repo-local pre-push hook (the global identity guard chains to it)
	@install -Dm755 tools/hooks/pre-push "$$(git rev-parse --absolute-git-dir)/hooks/pre-push"
	@echo "installed $$(git rev-parse --absolute-git-dir)/hooks/pre-push"
