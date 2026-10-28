# Thin wrappers over infra/scripts/solia.sh (which works without make, e.g. in Git Bash on Windows).
# Extra arguments go through ARGS: `make logs ARGS=backend`, `make reset ARGS=--yes`.
SOLIA := bash infra/scripts/solia.sh

.PHONY: help up down logs ps reset smoke dev-db

help:
	@$(SOLIA) help

up down logs ps reset smoke dev-db:
	@$(SOLIA) $@ $(ARGS)
