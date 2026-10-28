using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Npgsql;
using Solia.Shared.Configuration;

namespace Solia.DbMigrator;

/// <summary>
/// The migration run behind <c>Program.cs</c>, with its environment injected so tests can drive it
/// (exit codes, migrations-dir resolution) without touching the process environment.
/// </summary>
public static class Migrator
{
    public const int Success = 0;

    public const int Failure = 1;

    // Arbitrary constant shared by every runner instance; serializes concurrent migrations.
    public const long MigrationLockKey = 7_301_445_001;

    /// <summary>How long a run waits for another runner's lock by default, before failing instead of hanging.</summary>
    public static readonly TimeSpan DefaultLockTimeout = TimeSpan.FromMinutes(5);

    private static readonly TimeSpan LockPollInterval = TimeSpan.FromMilliseconds(500);

    private const string NoTransactionMarker = "-- solia:no-transaction";

    /// <summary>Applies the pending migrations; returns <see cref="Success"/> or <see cref="Failure"/>, never throws.</summary>
    public static async Task<int> RunAsync(
        string[] args, Func<string, string?> getEnv, TextWriter stdout, TextWriter stderr)
    {
        try
        {
            string connString = DbConnectionSettings.Resolve(getEnv);
            TimeSpan lockTimeout = ResolveLockTimeout(getEnv);
            string migrationsDir = ResolveMigrationsDir(
                args, getEnv, [Directory.GetCurrentDirectory(), AppContext.BaseDirectory]);

            string[] files = Directory.GetFiles(migrationsDir, "*.sql")
                .OrderBy(Path.GetFileName, StringComparer.Ordinal)
                .ToArray();

            stdout.WriteLine($"[migrator] migrations dir: {migrationsDir}");
            stdout.WriteLine($"[migrator] found {files.Length} migration file(s)");

            await using var conn = new NpgsqlConnection(connString);
            await conn.OpenAsync();

            // Session-level lock: held until released below or the connection closes.
            await AcquireLockAsync(conn, lockTimeout, stdout);
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
                        stdout.WriteLine($"[migrator] skip (already applied): {name}");
                        continue;
                    }

                    bool useTransaction = !sql.StartsWith(NoTransactionMarker, StringComparison.Ordinal);
                    stdout.WriteLine(useTransaction
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
                        // Not atomic: if the run dies between the statement and its record, the file is
                        // re-run next time — hence the re-runnable rule for such files.
                        try
                        {
                            await ExecuteAsync(conn, sql);
                            await RecordAppliedAsync(conn, name, checksum, tx: null);
                        }
                        catch (Exception)
                        {
                            stderr.WriteLine(
                                $"[migrator]   {name} runs outside a transaction and is re-run after an interrupted run, " +
                                "so it must be re-runnable (e.g. IF NOT EXISTS) — see services/db/README.md.");
                            throw;
                        }
                    }
                    appliedCount++;
                }

                stdout.WriteLine(appliedCount == 0
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
            return Success;
        }
        catch (PostgresException ex)
        {
            stderr.WriteLine($"[migrator] FAILED: {ex.SqlState} {ex.MessageText}");
            if (ex.Detail is not null) stderr.WriteLine($"[migrator]   detail: {ex.Detail}");
            if (ex.Hint is not null) stderr.WriteLine($"[migrator]   hint: {ex.Hint}");
            if (ex.Position > 0) stderr.WriteLine($"[migrator]   position: {ex.Position}");
            if (ex.Where is not null) stderr.WriteLine($"[migrator]   where: {ex.Where}");
            return Failure;
        }
        catch (Exception ex)
        {
            stderr.WriteLine($"[migrator] FAILED: {ex}");
            return Failure;
        }
    }

    /// <summary>
    /// Priority: CLI arg > <c>MIGRATIONS_DIR</c> env > probe upward from <paramref name="probeStarts"/> for the
    /// repo's <c>services/db/migrations</c>. An explicitly supplied path must exist — never silently fall back
    /// to probing (in the container image <c>MIGRATIONS_DIR</c> is set, so a broken image fails loudly).
    /// </summary>
    public static string ResolveMigrationsDir(
        string[] args, Func<string, string?> getEnv, IEnumerable<string> probeStarts)
    {
        string? explicitDir = args.Length > 0 ? args[0] : getEnv("MIGRATIONS_DIR");
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
        foreach (string start in probeStarts)
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

    /// <summary>
    /// <c>MIGRATOR_LOCK_TIMEOUT_SECONDS</c> (whole seconds, 0 = don't wait), else <see cref="DefaultLockTimeout"/>.
    /// </summary>
    public static TimeSpan ResolveLockTimeout(Func<string, string?> getEnv)
    {
        string? raw = getEnv("MIGRATOR_LOCK_TIMEOUT_SECONDS");
        if (string.IsNullOrWhiteSpace(raw))
        {
            return DefaultLockTimeout;
        }
        if (!int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out int seconds))
        {
            throw new InvalidOperationException(
                $"MIGRATOR_LOCK_TIMEOUT_SECONDS must be a non-negative whole number of seconds, got '{raw}'.");
        }
        return TimeSpan.FromSeconds(seconds);
    }

    // pg_try_advisory_lock in a bounded loop rather than pg_advisory_lock, which would wait forever
    // behind a stuck runner and never let the one-shot job exit.
    private static async Task AcquireLockAsync(NpgsqlConnection conn, TimeSpan timeout, TextWriter stdout)
    {
        var waited = Stopwatch.StartNew();
        await using var cmd = new NpgsqlCommand($"SELECT pg_try_advisory_lock({MigrationLockKey})", conn);
        while (!(bool)(await cmd.ExecuteScalarAsync())!)
        {
            if (waited.Elapsed >= timeout)
            {
                throw new TimeoutException(
                    $"Another migrator still holds the migration lock after {timeout.TotalSeconds:0}s " +
                    "(raise MIGRATOR_LOCK_TIMEOUT_SECONDS if a long migration is legitimately running).");
            }
            if (waited.Elapsed < LockPollInterval)
            {
                stdout.WriteLine("[migrator] waiting for another migrator to release the lock...");
            }
            await Task.Delay(LockPollInterval);
        }
    }

    private static async Task ExecuteAsync(NpgsqlConnection conn, string sql, NpgsqlTransaction? tx = null)
    {
        await using var cmd = new NpgsqlCommand(sql, conn, tx);
        await cmd.ExecuteNonQueryAsync();
    }

    private static async Task EnsureSchemaMigrationsTableAsync(NpgsqlConnection conn)
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

    private static async Task<Dictionary<string, string>> LoadAppliedAsync(NpgsqlConnection conn)
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

    private static async Task RecordAppliedAsync(NpgsqlConnection conn, string name, string checksum, NpgsqlTransaction? tx)
    {
        await using var cmd = new NpgsqlCommand(
            "INSERT INTO schema_migrations (filename, checksum) VALUES (@f, @c)", conn, tx);
        cmd.Parameters.AddWithValue("f", name);
        cmd.Parameters.AddWithValue("c", checksum);
        await cmd.ExecuteNonQueryAsync();
    }

    // Checksums must not depend on the checkout's line endings (CRLF on Windows, LF elsewhere).
    private static string NormalizeLineEndings(string text) => text.Replace("\r\n", "\n").Replace('\r', '\n');

    private static string Sha256Hex(string text)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
