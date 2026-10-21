// Solia.DbMigrator — minimal, dependency-light SQL migration runner (Sprint 1 / T1).
//
// Applies ordered *.sql files from the migrations directory exactly once, tracking
// applied files (with a SHA-256 checksum) in a schema_migrations table. Re-running is a
// no-op, and concurrent runners are serialized by a Postgres advisory lock, so it is safe
// to run on every container start (Sprint 6).
//
// Configuration (environment variables):
//   SOLIA_DB_CONNECTION  full Npgsql connection string (takes precedence), OR
//   POSTGRES_HOST (default "localhost"), POSTGRES_PORT (5432), POSTGRES_DB (solia),
//   POSTGRES_USER (solia), POSTGRES_PASSWORD (change_me)
//   MIGRATIONS_DIR       path to the migrations folder (argv[0] takes precedence; an
//                        explicit path that does not exist is an error)
//
// A migration whose first line is "-- solia:no-transaction" runs outside a transaction
// (needed for e.g. CREATE INDEX CONCURRENTLY); such a file must hold a single statement.
//
// Exit codes: 0 success (including "nothing to apply"), 1 failure.

using System.Security.Cryptography;
using System.Text;
using Npgsql;

// Arbitrary constant shared by every runner instance; serializes concurrent migrations.
const long MigrationLockKey = 7_301_445_001;
const string NoTransactionMarker = "-- solia:no-transaction";

try
{
    string connString = BuildConnectionString();
    string migrationsDir = ResolveMigrationsDir(args);

    string[] files = Directory.GetFiles(migrationsDir, "*.sql")
        .OrderBy(Path.GetFileName, StringComparer.Ordinal)
        .ToArray();

    Console.WriteLine($"[migrator] migrations dir: {migrationsDir}");
    Console.WriteLine($"[migrator] found {files.Length} migration file(s)");

    await using var conn = new NpgsqlConnection(connString);
    await conn.OpenAsync();

    // Session-level lock: held until released below or the connection closes.
    await ExecuteAsync(conn, $"SELECT pg_advisory_lock({MigrationLockKey})");
    try
    {
        await EnsureSchemaMigrationsTableAsync(conn);
        Dictionary<string, string> applied = await LoadAppliedAsync(conn);

        int appliedCount = 0;
        foreach (string file in files)
        {
            string name = Path.GetFileName(file);
            string sql = NormalizeLineEndings(await File.ReadAllTextAsync(file));
            string checksum = Sha256Hex(sql);

            if (applied.TryGetValue(name, out string? knownChecksum))
            {
                if (knownChecksum != checksum)
                {
                    throw new InvalidOperationException(
                        $"Checksum drift for already-applied migration '{name}'. " +
                        "An applied migration file must never be edited — add a new migration instead.");
                }
                Console.WriteLine($"[migrator] skip (already applied): {name}");
                continue;
            }

            bool useTransaction = !sql.StartsWith(NoTransactionMarker, StringComparison.Ordinal);
            Console.WriteLine(useTransaction
                ? $"[migrator] applying: {name}"
                : $"[migrator] applying (no transaction): {name}");

            if (useTransaction)
            {
                // Disposing an uncommitted transaction rolls it back.
                await using NpgsqlTransaction tx = await conn.BeginTransactionAsync();
                await ExecuteAsync(conn, sql, tx);
                await RecordAppliedAsync(conn, name, checksum, tx);
                await tx.CommitAsync();
            }
            else
            {
                await ExecuteAsync(conn, sql);
                await RecordAppliedAsync(conn, name, checksum, tx: null);
            }
            appliedCount++;
        }

        Console.WriteLine(appliedCount == 0
            ? "[migrator] up to date — nothing to apply."
            : $"[migrator] done — applied {appliedCount} migration(s).");
    }
    finally
    {
        // Best effort: the lock is released with the session anyway, and an unlock error
        // must not mask the original failure.
        try
        {
            await ExecuteAsync(conn, $"SELECT pg_advisory_unlock({MigrationLockKey})");
        }
        catch (Exception)
        {
        }
    }
    return 0;
}
catch (PostgresException ex)
{
    Console.Error.WriteLine($"[migrator] FAILED: {ex.SqlState} {ex.MessageText}");
    if (ex.Detail is not null) Console.Error.WriteLine($"[migrator]   detail: {ex.Detail}");
    if (ex.Hint is not null) Console.Error.WriteLine($"[migrator]   hint: {ex.Hint}");
    if (ex.Position > 0) Console.Error.WriteLine($"[migrator]   position: {ex.Position}");
    if (ex.Where is not null) Console.Error.WriteLine($"[migrator]   where: {ex.Where}");
    return 1;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"[migrator] FAILED: {ex}");
    return 1;
}

static string BuildConnectionString()
{
    string? explicitConn = Environment.GetEnvironmentVariable("SOLIA_DB_CONNECTION");
    if (!string.IsNullOrWhiteSpace(explicitConn))
    {
        return explicitConn;
    }

    var builder = new NpgsqlConnectionStringBuilder
    {
        Host = Env("POSTGRES_HOST", "localhost"),
        Port = int.Parse(Env("POSTGRES_PORT", "5432")),
        Database = Env("POSTGRES_DB", "solia"),
        Username = Env("POSTGRES_USER", "solia"),
        Password = Env("POSTGRES_PASSWORD", "change_me"),
    };
    return builder.ConnectionString;
}

static string Env(string key, string fallback)
{
    string? v = Environment.GetEnvironmentVariable(key);
    return string.IsNullOrWhiteSpace(v) ? fallback : v;
}

static string ResolveMigrationsDir(string[] args)
{
    // Priority: CLI arg > MIGRATIONS_DIR env > probe for the repo's services/db/migrations.
    // An explicitly supplied path must exist — never silently fall back to probing.
    string? explicitDir = args.Length > 0 ? args[0] : Environment.GetEnvironmentVariable("MIGRATIONS_DIR");
    if (!string.IsNullOrWhiteSpace(explicitDir))
    {
        if (!Directory.Exists(explicitDir))
        {
            throw new DirectoryNotFoundException($"Migrations directory '{explicitDir}' does not exist.");
        }
        return Path.GetFullPath(explicitDir);
    }

    // Only the repo layout is probed (no bare "migrations" folders, which other
    // services/tools may also have).
    foreach (string start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
    {
        DirectoryInfo? dir = new(start);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "services", "db", "migrations");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
    }

    throw new DirectoryNotFoundException(
        "Could not locate services/db/migrations. Pass the directory as the first argument or set MIGRATIONS_DIR.");
}

static async Task ExecuteAsync(NpgsqlConnection conn, string sql, NpgsqlTransaction? tx = null)
{
    await using var cmd = new NpgsqlCommand(sql, conn, tx);
    await cmd.ExecuteNonQueryAsync();
}

static async Task EnsureSchemaMigrationsTableAsync(NpgsqlConnection conn)
{
    const string sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename   TEXT        PRIMARY KEY,
            checksum   TEXT        NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """;
    await ExecuteAsync(conn, sql);
}

static async Task<Dictionary<string, string>> LoadAppliedAsync(NpgsqlConnection conn)
{
    var result = new Dictionary<string, string>(StringComparer.Ordinal);
    await using var cmd = new NpgsqlCommand("SELECT filename, checksum FROM schema_migrations", conn);
    await using NpgsqlDataReader reader = await cmd.ExecuteReaderAsync();
    while (await reader.ReadAsync())
    {
        result[reader.GetString(0)] = reader.GetString(1);
    }
    return result;
}

static async Task RecordAppliedAsync(NpgsqlConnection conn, string name, string checksum, NpgsqlTransaction? tx)
{
    await using var cmd = new NpgsqlCommand(
        "INSERT INTO schema_migrations (filename, checksum) VALUES (@f, @c)", conn, tx);
    cmd.Parameters.AddWithValue("f", name);
    cmd.Parameters.AddWithValue("c", checksum);
    await cmd.ExecuteNonQueryAsync();
}

// Checksums must not depend on the checkout's line endings (CRLF on Windows, LF elsewhere).
static string NormalizeLineEndings(string text) => text.Replace("\r\n", "\n").Replace('\r', '\n');

static string Sha256Hex(string text)
{
    byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(text));
    return Convert.ToHexString(hash).ToLowerInvariant();
}
