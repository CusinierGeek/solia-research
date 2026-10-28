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
}
