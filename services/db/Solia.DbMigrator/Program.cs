// Solia.DbMigrator — minimal, dependency-light SQL migration runner (Sprint 1 / T1).
//
// Applies ordered *.sql files from the migrations directory exactly once, tracking
// applied files (with a SHA-256 checksum) in a schema_migrations table. Re-running is a
// no-op, and concurrent runners are serialized by a Postgres advisory lock, so it is safe
// to run on every container start: in Docker it is a one-shot job
// (infra/docker/db-migrator.Dockerfile) that the app services wait on.
//
// Configuration (environment variables):
//   SOLIA_DB_CONNECTION  full Npgsql connection string (takes precedence), OR
//   POSTGRES_HOST (default "localhost"), POSTGRES_PORT (5432), POSTGRES_DB (solia),
//   POSTGRES_USER (solia), POSTGRES_PASSWORD (change_me)
//   MIGRATIONS_DIR       path to the migrations folder (argv[0] takes precedence; an
//                        explicit path that does not exist is an error)
//   MIGRATOR_LOCK_TIMEOUT_SECONDS  how long to wait for another runner's lock (default 300)
//
// A migration whose first line is "-- solia:no-transaction" runs outside a transaction
// (needed for e.g. CREATE INDEX CONCURRENTLY); such a file must hold a single statement.
//
// Exit codes: 0 success (including "nothing to apply"), 1 failure (bad SQL, checksum
// drift, missing migrations dir, unreachable database, lock wait timed out).

using Solia.DbMigrator;

return await Migrator.RunAsync(args, Environment.GetEnvironmentVariable, Console.Out, Console.Error);
