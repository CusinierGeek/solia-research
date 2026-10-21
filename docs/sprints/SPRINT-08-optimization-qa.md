# Sprint 8 — Optimization & QA

**Milestone:** Optimization & QA · **Depends on:** all · **Service:** cross-cutting

## Goal
Harden performance, reliability, observability, and test coverage; prepare for sustained operation.

## Scope
- DB: indexes review, partitioning/TTL for `features`/`logs`/`predictions` (retention in DATA-MODEL).
- Pipeline: throughput, batching, backpressure under load.
- AI: inference latency (batching, TorchScript/ONNX optional), training time.
- Backend/dashboard: caching, pagination limits, N+1 checks.
- Observability: metrics endpoints, dashboards, alerting hooks.
- CI hardening: coverage gates, image publish, security scan.

## Task breakdown
- [ ] Load-test the pipeline; tune windows/batching.
- [ ] Add DB indexes/partitions + retention jobs.
- [ ] Benchmark + optimize inference latency; consider ONNX export.
- [ ] Add metrics (Prometheus-style) per service + a health/metrics dashboard.
- [ ] Review and raise test coverage; add end-to-end smoke test.
- [ ] CI: coverage threshold, container build/publish, dependency/security scan.
- [ ] Performance report + tuning notes doc.

## Deliverables
- Measurable performance improvements with before/after numbers.
- Retention/partitioning in place.
- Metrics + coverage gates in CI.
- `docs/PERFORMANCE.md` tuning notes.

## Definition of Done
- [ ] Documented latency/throughput targets met.
- [ ] Retention keeps table sizes bounded.
- [ ] CI enforces coverage and builds/publishes images.
- [ ] End-to-end smoke test green.

## Kickoff prompt
> We're implementing **Sprint 8 — Optimization & QA** of Solia. Read `docs/ARCHITECTURE.md` and `docs/DATA-MODEL.md` (retention section). Load-test and tune the data pipeline, add DB indexes/partitioning and retention jobs for high-volume tables, benchmark and optimize AI inference latency (consider ONNX), add per-service metrics and a metrics dashboard, raise test coverage with an end-to-end smoke test, and harden CI (coverage gate, image publish, security scan). Produce `docs/PERFORMANCE.md` with before/after numbers. Branch `sprint/08-optimization-qa`, PR per `CONTRIBUTING.md`.
