# data-pipeline (.NET 10 worker)

Ingests Pump.fun activity from **pumpapi.io** — live WebSocket **and** historical replay — normalizes events, computes features across the full token lifecycle (pre/post-bonding), and persists them to PostgreSQL.

- **Implemented in:** [Sprint 1](../../docs/sprints/SPRINT-01-data-pipeline.md)
- **Design:** [DATA-SOURCES.md](../../docs/DATA-SOURCES.md) · [DATA-PIPELINE.md](../../docs/DATA-PIPELINE.md)

Planned project: `Solia.DataPipeline` (worker). The autonomous `SolanaRpcSource` fallback reuses the Pump.fun IDL decoder from the old SolGemFinder project.

_Empty placeholder — code lands in Sprint 1._
