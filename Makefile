# Vendored Layer-A shared installer core (installkit).
#
# installkit is DEVELOPED in its own repo and VENDORED into this tree at a pinned commit, per the locked
# packaging model (INSTALL_PLAN §4/§9). The app imports `installkit` from the vendored copy at the repo
# root (installkit/); nothing depends on the sibling checkout at runtime, and the Pi rsync ships the
# vendored copy because it is git-tracked (`git ls-files '*.py'`).
#
# PROVISIONAL PIN (push freeze): installkit is not yet on its public origin, so INSTALLKIT_PIN is a LOCAL
# commit today and `vendor-check` compares the vendored bytes against the local installkit repo. When the maintainer
# lifts the freeze and pushes installkit, the same SHA is the public pin and the check runs unchanged
# against the public origin. This is deliberately NOT wired into pytest/CI as a blocking gate until then.

INSTALLKIT_PIN ?= 78ff1cd
INSTALLKIT_SRC ?= $(HOME)/dev/installkit
VENDOR_DIR     := installkit
MODULES        := __init__.py deps.py hardware.py secrets.py templating.py wizard.py

.PHONY: vendor-sync vendor-check

vendor-sync:   ## Re-copy installkit modules from $(INSTALLKIT_SRC) at $(INSTALLKIT_PIN) into $(VENDOR_DIR)/
	@for m in $(MODULES); do \
	  git -C "$(INSTALLKIT_SRC)" show "$(INSTALLKIT_PIN):installkit/$$m" > "$(VENDOR_DIR)/$$m" \
	    && echo "vendored $$m @ $(INSTALLKIT_PIN)"; \
	done
	@echo "Done. Review 'git diff $(VENDOR_DIR)/' and commit (maintainer-gated)."

vendor-check:  ## Fail if any vendored module diverges from installkit@$(INSTALLKIT_PIN) (content-derive SHA-match)
	@rc=0; for m in $(MODULES); do \
	  if git -C "$(INSTALLKIT_SRC)" show "$(INSTALLKIT_PIN):installkit/$$m" | cmp -s - "$(VENDOR_DIR)/$$m"; then \
	    echo "OK    $$m"; \
	  else \
	    echo "DRIFT $$m  (vendored copy != installkit@$(INSTALLKIT_PIN))"; rc=1; \
	  fi; \
	done; \
	[ $$rc -eq 0 ] || echo "vendor-check FAILED — run 'make vendor-sync' or reconcile the pin."; \
	exit $$rc
