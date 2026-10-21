# Contributing & PR process

## Workflow model

```
Sprint = GitHub Milestone = a set of Tasks
Task   = GitHub Issue      = exactly ONE pull request
```

- Tasks are specified in their **GitHub Issue** (source of truth) using the **Task** issue template: *Business need · Acceptance criteria · Technical details · Dependencies · Out of scope*.
- The `docs/sprints/SPRINT-0N-*.md` files are the sprint **brief** only; per-task detail lives in issues.
- **One task = one PR.** Keep PRs scoped to a single issue.
- All tasks for a sprint are tracked on the **Project (v2) "Solia"** board and under the sprint's **Milestone**.

## Branching model

- `main` is **protected** and always deployable. No direct pushes.
- Work happens on branches named by sprint/type:
  - `sprint/01-data-pipeline`, `sprint/02-ai-model`, …
  - `feat/<short-desc>`, `fix/<short-desc>`, `docs/<short-desc>`, `chore/<short-desc>`

## Commit convention

[Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): summary` — e.g. `feat(pipeline): add PumpPortal WS reconnect`.
Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `ci`.

## Pull request flow

1. Branch off `main`.
2. Implement; keep the PR scoped to one sprint/task.
3. Ensure **lint + tests pass locally** and the sprint **Definition of Done** is met.
4. Open a PR into `main` using the PR template.
5. Link the PR to its **milestone** (the sprint) and relevant issues (`Closes #N`).
6. CI must be green. At least **one approving review** is required before merge.
7. Prefer **Squash and merge**; delete the branch after merge.

## Review checklist (reviewer)

- [ ] Matches the sprint scope & Definition of Done.
- [ ] Contracts (`API-CONTRACTS.md`, `DATA-MODEL.md`) respected; changes to them are intentional and documented.
- [ ] Tests cover new logic; CI green.
- [ ] No secrets committed; no decrypted secret logged.
- [ ] **No real-money/trade-execution path** — paper trading only.
- [ ] Clear naming, comments where non-obvious, no dead code.

## Local checks before pushing

- Python services: `ruff check . && black --check . && pytest`
- .NET services: `dotnet format --verify-no-changes && dotnet test`

## Definition of Done (global)

A PR is done when: code + tests written, CI green, docs updated if contracts changed, the sprint checklist items it claims are checked, and a reviewer approved.

## Issues & milestones

- One **milestone per sprint** (mirrors `docs/ROADMAP.md`).
- Each sprint's work is broken into **Task issues** (template: *Task*), labelled `sprint-<n>` + `type:task`, assigned to the sprint milestone, and added to the **Solia** Project board.
- Each task issue becomes one PR that `Closes #<issue>`.
