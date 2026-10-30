using Npgsql;
using Solia.DbMigrator;

namespace Solia.DbMigrator.Tests;

/// <summary>The shared Postgres container, left empty: the migrator under test applies the schema.</summary>
public sealed class EmptyPostgresFixture() : PostgresFixture(applyMigrations: false)
{
    private int _databases;

    /// <summary>A fresh, empty database on the shared container; returns its connection string.</summary>
    public async Task<string> CreateDatabaseAsync()
    {
        string name = $"m{Interlocked.Increment(ref _databases)}";
        await using (NpgsqlCommand cmd = DataSource.CreateCommand($"CREATE DATABASE {name}"))
        {
            await cmd.ExecuteNonQueryAsync();
        }
        return new NpgsqlConnectionStringBuilder(ConnectionString) { Database = name }.ConnectionString;
    }
}

[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class MigratorIntegrationTests(EmptyPostgresFixture db) : IClassFixture<EmptyPostgresFixture>, IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("solia-migrations-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static async Task<(int Code, string Out, string Err)> RunAsync(
        string connString, string migrationsDir, string? lockTimeoutSeconds = null)
    {
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        int code = await Migrator.RunAsync(
            [], key => key switch
            {
                "SOLIA_DB_CONNECTION" => connString,
                "MIGRATIONS_DIR" => migrationsDir,
                "MIGRATOR_LOCK_TIMEOUT_SECONDS" => lockTimeoutSeconds,
                _ => null,
            },
            stdout, stderr);
        return (code, stdout.ToString(), stderr.ToString());
    }

    private static async Task<T> ScalarAsync<T>(string connString, string sql)
    {
        await using var dataSource = NpgsqlDataSource.Create(connString);
        return await PostgresFixture.ScalarAsync<T>(dataSource, sql);
    }

    private static async Task ExecuteAsync(string connString, string sql)
    {
        await using var dataSource = NpgsqlDataSource.Create(connString);
        await using NpgsqlCommand cmd = dataSource.CreateCommand(sql);
        await cmd.ExecuteNonQueryAsync();
    }

    [Fact]
    public async Task RepoMigrations_ApplyOnAnEmptyDatabase_ThenSecondRunIsANoOp()
    {
        string conn = await db.CreateDatabaseAsync();
        string repoMigrations = PostgresFixture.FindMigrationsDir();
        int fileCount = Directory.GetFiles(repoMigrations, "*.sql").Length;

        var first = await RunAsync(conn, repoMigrations);

        Assert.True(first.Code == Migrator.Success, first.Err);
        Assert.Equal(fileCount, await ScalarAsync<long>(conn, "SELECT count(*) FROM schema_migrations"));
        Assert.NotNull(await ScalarAsync<string>(conn, "SELECT to_regclass('public.tokens_live')::text"));

        var second = await RunAsync(conn, repoMigrations);

        Assert.True(second.Code == Migrator.Success, second.Err);
        Assert.Contains("nothing to apply", second.Out);
        Assert.Equal(fileCount, await ScalarAsync<long>(conn, "SELECT count(*) FROM schema_migrations"));
    }

    [Fact]
    public async Task ConcurrentRuns_AreSerialized_EachFileAppliedOnce()
    {
        string conn = await db.CreateDatabaseAsync();
        // Not idempotent on purpose: a second, unserialized apply would fail with "already exists".
        File.WriteAllText(Path.Combine(_dir, "0001_t.sql"), "CREATE TABLE t (id INT); SELECT pg_sleep(0.5);");
        File.WriteAllText(Path.Combine(_dir, "0002_u.sql"), "CREATE TABLE u (id INT);");

        var results = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => RunAsync(conn, _dir)));

        Assert.All(results, r => Assert.True(r.Code == Migrator.Success, r.Err));
        Assert.Equal(2L, await ScalarAsync<long>(conn, "SELECT count(*) FROM schema_migrations"));
        Assert.Single(results, r => r.Out.Contains("applied 2 migration(s)"));
    }

    [Fact]
    public async Task LockHeldByAnotherSession_ExitsNonZero_InsteadOfWaitingForever()
    {
        string conn = await db.CreateDatabaseAsync();
        File.WriteAllText(Path.Combine(_dir, "0001_t.sql"), "CREATE TABLE t (id INT);");
        await using var holder = new NpgsqlConnection(conn);
        await holder.OpenAsync();
        await using var lockCmd = new NpgsqlCommand($"SELECT pg_advisory_lock({Migrator.MigrationLockKey})", holder);
        await lockCmd.ExecuteNonQueryAsync();

        var result = await RunAsync(conn, _dir, lockTimeoutSeconds: "1").WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(Migrator.Failure, result.Code);
        Assert.Contains("still holds the migration lock", result.Err);
        Assert.Null(await ScalarAsync<string>(conn, "SELECT to_regclass('public.t')::text"));

        // Once the holder releases it, the next run goes through.
        lockCmd.CommandText = $"SELECT pg_advisory_unlock({Migrator.MigrationLockKey})";
        await lockCmd.ExecuteNonQueryAsync();
        var retry = await RunAsync(conn, _dir, lockTimeoutSeconds: "1");
        Assert.True(retry.Code == Migrator.Success, retry.Err);
    }

    [Fact]
    public async Task NoTransactionMigration_AppliedButNotRecorded_IsRecordedOnTheNextRun()
    {
        // Simulates a run killed between a no-transaction statement and its schema_migrations row.
        string conn = await db.CreateDatabaseAsync();
        File.WriteAllText(Path.Combine(_dir, "0001_t.sql"), "CREATE TABLE t (id INT);");
        Assert.Equal(Migrator.Success, (await RunAsync(conn, _dir)).Code);
        File.WriteAllText(
            Path.Combine(_dir, "0002_idx.sql"),
            "-- solia:no-transaction\nCREATE INDEX CONCURRENTLY IF NOT EXISTS t_id ON t (id);\n");
        await ExecuteAsync(conn, "CREATE INDEX CONCURRENTLY t_id ON t (id)");

        var result = await RunAsync(conn, _dir);

        Assert.True(result.Code == Migrator.Success, result.Err);
        Assert.Equal(2L, await ScalarAsync<long>(conn, "SELECT count(*) FROM schema_migrations"));
    }

    [Fact]
    public async Task NoTransactionMigration_ThatIsNotReRunnable_FailsWithAHint()
    {
        string conn = await db.CreateDatabaseAsync();
        File.WriteAllText(Path.Combine(_dir, "0001_t.sql"), "CREATE TABLE t (id INT); CREATE INDEX t_id ON t (id);");
        File.WriteAllText(
            Path.Combine(_dir, "0002_idx.sql"), "-- solia:no-transaction\nCREATE INDEX CONCURRENTLY t_id ON t (id);\n");

        var result = await RunAsync(conn, _dir);

        Assert.Equal(Migrator.Failure, result.Code);
        Assert.Contains("42P07", result.Err); // duplicate_table (relation already exists)
        Assert.Contains("must be re-runnable", result.Err);
        Assert.Equal(1L, await ScalarAsync<long>(conn, "SELECT count(*) FROM schema_migrations"));
    }

    [Fact]
    public async Task BadSql_ExitsNonZero_AndRollsBackThatMigration()
    {
        string conn = await db.CreateDatabaseAsync();
        File.WriteAllText(Path.Combine(_dir, "0001_ok.sql"), "CREATE TABLE ok (id INT);");
        File.WriteAllText(Path.Combine(_dir, "0002_bad.sql"), "CREATE TABLE half (id INT); SELEC broken;");

        var result = await RunAsync(conn, _dir);

        Assert.Equal(Migrator.Failure, result.Code);
        Assert.Contains("42601", result.Err); // syntax_error SQLSTATE
        Assert.Equal("0001_ok.sql", await ScalarAsync<string>(conn, "SELECT string_agg(filename, ',') FROM schema_migrations"));
        Assert.Null(await ScalarAsync<string>(conn, "SELECT to_regclass('public.half')::text"));
    }

    [Fact]
    public async Task EditedAppliedMigration_ExitsNonZero_OnChecksumDrift()
    {
        string conn = await db.CreateDatabaseAsync();
        string file = Path.Combine(_dir, "0001_t.sql");
        File.WriteAllText(file, "CREATE TABLE t (id INT);");
        Assert.Equal(Migrator.Success, (await RunAsync(conn, _dir)).Code);

        File.WriteAllText(file, "CREATE TABLE t (id BIGINT);");
        var result = await RunAsync(conn, _dir);

        Assert.Equal(Migrator.Failure, result.Code);
        Assert.Contains("Checksum drift", result.Err);
    }

    [Fact]
    public async Task LineEndings_DoNotChangeTheChecksum()
    {
        string conn = await db.CreateDatabaseAsync();
        string file = Path.Combine(_dir, "0001_t.sql");
        File.WriteAllText(file, "CREATE TABLE t (id INT);\nSELECT 1;\n");
        Assert.Equal(Migrator.Success, (await RunAsync(conn, _dir)).Code);

        File.WriteAllText(file, "CREATE TABLE t (id INT);\r\nSELECT 1;\r\n");
        var result = await RunAsync(conn, _dir);

        Assert.True(result.Code == Migrator.Success, result.Err);
        Assert.Contains("nothing to apply", result.Out);
    }

    [Fact]
    public async Task BurnedLiquidity_IsBackfilledInBatches_AndFilledForWritersThatOmitIt()
    {
        // FE-T5 (#114): 0008 adds the column + a fill-on-insert trigger, 0009 backfills in batches.
        string conn = await db.CreateDatabaseAsync();
        string[] repo = Directory.GetFiles(PostgresFixture.FindMigrationsDir(), "*.sql").Order().ToArray();
        foreach (string file in repo.Where(f => string.CompareOrdinal(Path.GetFileName(f), "0008") < 0))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }
        Assert.Equal(Migrator.Success, (await RunAsync(conn, _dir)).Code);

        // Rows written before 0008, ids far enough apart to span several batches.
        await ExecuteAsync(conn, """
            INSERT INTO tokens_history (mint) VALUES ('M');
            INSERT INTO features (id, mint, ts, onchain_signals) VALUES
                (1,      'M', '2026-01-01T00:00:01Z', '{"burnedLiquidityPercent": 57.5}'),
                (120000, 'M', '2026-01-01T00:00:02Z', '{"burnedLiquidityPercent": 0}'),
                (120001, 'M', '2026-01-01T00:00:03Z', '{"burnedLiquidityPercent": null}'),
                (250000, 'M', '2026-01-01T00:00:04Z', '{"burnedLiquidityPercent": "lots"}');
            """);
        foreach (string file in repo.Where(f => string.CompareOrdinal(Path.GetFileName(f), "0008") >= 0))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }

        var result = await RunAsync(conn, _dir);

        Assert.True(result.Code == Migrator.Success, result.Err);
        Assert.Equal("1=57.5;120000=0;120001=;250000=", await ScalarAsync<string>(conn,
            "SELECT string_agg(id || '=' || coalesce(burned_liquidity::text, ''), ';' ORDER BY id) FROM features"));

        // A writer that leaves the column out (a pipeline older than 0008) gets it from the JSONB;
        // one that writes it keeps its value.
        await ExecuteAsync(conn, """
            INSERT INTO features (id, mint, ts, onchain_signals) VALUES (300000, 'M', '2026-01-01T00:00:05Z', '{"burnedLiquidityPercent": 100}');
            INSERT INTO features (id, mint, ts, burned_liquidity, onchain_signals) VALUES (300001, 'M', '2026-01-01T00:00:06Z', 42, '{"burnedLiquidityPercent": 100}');
            """);
        Assert.Equal("300000=100;300001=42", await ScalarAsync<string>(conn,
            "SELECT string_agg(id || '=' || burned_liquidity, ';' ORDER BY id) FROM features WHERE id >= 300000"));
    }

    [Fact]
    public async Task TrainingEvents_AcceptManualAndFailed_OnTopOf0009_AndStillRejectUnknownValues()
    {
        // S7-T2 (#130): 0010 widens the trigger / decision CHECKs of a database already at 0009, backfills the rows
        // whose decision and finished_at disagree, then requires both to be set together.
        string conn = await db.CreateDatabaseAsync();
        string[] repo = Directory.GetFiles(PostgresFixture.FindMigrationsDir(), "*.sql").Order().ToArray();
        foreach (string file in repo.Where(f => string.CompareOrdinal(Path.GetFileName(f), "0010") < 0))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }
        Assert.Equal(Migrator.Success, (await RunAsync(conn, _dir)).Code);
        await ExecuteAsync(conn, """
            INSERT INTO training_events (id, started_at, finished_at, trigger, decision) VALUES
                (900, '2026-01-01T00:00:00Z', NULL, 'n_trades', 'promoted'),
                (901, '2026-01-02T00:00:00Z', '2026-01-02T01:00:00Z', 'initial', NULL),
                (902, '2026-01-03T00:00:00Z', NULL, 'scheduled', NULL);
            """);
        var before = await Assert.ThrowsAsync<PostgresException>(() =>
            ExecuteAsync(conn, "INSERT INTO training_events (trigger) VALUES ('manual')"));
        Assert.Equal(PostgresErrorCodes.CheckViolation, before.SqlState);

        foreach (string file in repo.Where(f => string.CompareOrdinal(Path.GetFileName(f), "0010") >= 0))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }
        var result = await RunAsync(conn, _dir);
        Assert.True(result.Code == Migrator.Success, result.Err);

        // A decided run without finished_at is closed at its start; a finished one without a decision has no known
        // outcome, so it becomes failed; an in-flight one is left alone.
        Assert.Equal("900:promoted:2026-01-01;901:failed:2026-01-02;902::", await ScalarAsync<string>(conn,
            "SELECT string_agg(id || ':' || coalesce(decision, '') || ':' || coalesce(to_char(finished_at AT TIME ZONE 'UTC', 'YYYY-MM-DD'), ''), ';' ORDER BY id) " +
            "FROM training_events WHERE id >= 900"));
        await ExecuteAsync(conn, "DELETE FROM training_events WHERE id >= 900");

        await ExecuteAsync(conn, """
            INSERT INTO training_events (trigger, finished_at, decision) VALUES
                ('manual', now(), 'failed'), ('initial', now(), 'rejected'), ('scheduled', now(), 'rollback'),
                ('n_trades', NULL, NULL);
            """);
        Assert.Equal("initial,manual,n_trades,scheduled", await ScalarAsync<string>(conn,
            "SELECT string_agg(trigger, ',' ORDER BY trigger) FROM training_events"));

        foreach (string bad in new[]
        {
            "INSERT INTO training_events (trigger) VALUES ('cron')",
            "INSERT INTO training_events (decision) VALUES ('error')",
            "INSERT INTO training_events (trigger) VALUES ('Manual')",
            "INSERT INTO training_events (trigger, decision) VALUES ('n_trades', 'promoted')",
            "INSERT INTO training_events (trigger, finished_at) VALUES ('n_trades', now())",
        })
        {
            var ex = await Assert.ThrowsAsync<PostgresException>(() => ExecuteAsync(conn, bad));
            Assert.Equal(PostgresErrorCodes.CheckViolation, ex.SqlState);
        }
    }

    [Fact]
    public async Task BotSessions_BackfillSplitsTheTradesAtTheRealisticFillsCutover()
    {
        // #169: 0012 puts every trade opened before 2026-10-30 20:26:42 UTC in a closed "naive fills" session and every
        // later one in the current "realistic fills" session, each with the champion most of its trades came from.
        string conn = await db.CreateDatabaseAsync();
        string[] repo = Directory.GetFiles(PostgresFixture.FindMigrationsDir(), "*.sql").Order().ToArray();
        foreach (string file in repo.Where(f => string.CompareOrdinal(Path.GetFileName(f), "0012") < 0))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }
        Assert.Equal(Migrator.Success, (await RunAsync(conn, _dir)).Code);
        await ExecuteAsync(conn, """
            INSERT INTO tokens_history (mint) VALUES ('M');
            INSERT INTO models (id, version, path, is_champion) VALUES (1, 'v1', 'models/v1.pt', TRUE);
            INSERT INTO predictions (id, mint, model_id, ts, horizon) VALUES (10, 'M', 1, '2026-10-30T14:00:00Z', '15m');
            INSERT INTO trades (id, mint, prediction_id, opened_at, closed_at, pnl, status) VALUES
                (1, 'M', 10,   '2026-10-30T14:06:10Z', '2026-10-30T14:10:00Z', 1, 'closed'),
                (2, 'M', NULL, '2026-10-30T20:15:04Z', NULL, NULL, 'open'),
                (3, 'M', 10,   '2026-10-30T20:26:42Z', '2026-10-30T20:30:00Z', -1, 'closed');
            """);
        foreach (string file in repo.Where(f => string.CompareOrdinal(Path.GetFileName(f), "0012") >= 0))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }

        var result = await RunAsync(conn, _dir);

        Assert.True(result.Code == Migrator.Success, result.Err);
        Assert.Equal(
            "naive fills (pre-#165)|2026-10-30 14:06:10+00|2026-10-30 20:26:42+00|v1;realistic fills (#165)|2026-10-30 20:26:42+00||v1",
            await ScalarAsync<string>(conn, """
                SELECT string_agg(concat_ws('|', label, started_at::text, coalesce(ended_at::text, ''), model_version), ';' ORDER BY id)
                FROM bot_sessions
                """));
        Assert.Equal("1=naive fills (pre-#165);2=naive fills (pre-#165);3=realistic fills (#165)", await ScalarAsync<string>(conn,
            "SELECT string_agg(t.id || '=' || s.label, ';' ORDER BY t.id) FROM trades t JOIN bot_sessions s ON s.id = t.session_id"));
        Assert.Null(await ScalarAsync<string>(conn, "SELECT config_hash FROM bot_sessions WHERE config_hash IS NOT NULL LIMIT 1"));

        // At most one current session.
        var second = await Assert.ThrowsAsync<PostgresException>(() => ExecuteAsync(conn, "INSERT INTO bot_sessions DEFAULT VALUES"));
        Assert.Equal(PostgresErrorCodes.UniqueViolation, second.SqlState);
    }

    [Fact]
    public async Task BotSessions_BackfillOnADatabaseWithoutTradesInsertsNothing()
    {
        string conn = await db.CreateDatabaseAsync();

        var result = await RunAsync(conn, PostgresFixture.FindMigrationsDir());

        Assert.True(result.Code == Migrator.Success, result.Err);
        Assert.Equal(0L, await ScalarAsync<long>(conn, "SELECT count(*) FROM bot_sessions"));
    }
}
