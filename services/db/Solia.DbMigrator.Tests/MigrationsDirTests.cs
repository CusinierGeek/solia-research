using Solia.DbMigrator;

namespace Solia.DbMigrator.Tests;

public sealed class MigrationsDirTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("solia-migrator-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static Func<string, string?> Env(params (string Key, string Value)[] vars) =>
        key => vars.FirstOrDefault(v => v.Key == key).Value;

    [Fact]
    public void MigrationsDirEnv_IsUsed_AsInTheImage()
    {
        // The image sets MIGRATIONS_DIR=/app/migrations; the probe roots are not under any repo.
        string dir = Directory.CreateDirectory(Path.Combine(_root, "app", "migrations")).FullName;

        string resolved = Migrator.ResolveMigrationsDir([], Env(("MIGRATIONS_DIR", dir)), [_root]);

        Assert.Equal(dir, resolved);
    }

    [Fact]
    public void CliArg_TakesPrecedenceOverEnv()
    {
        string fromArg = Directory.CreateDirectory(Path.Combine(_root, "arg")).FullName;
        string fromEnv = Directory.CreateDirectory(Path.Combine(_root, "env")).FullName;

        string resolved = Migrator.ResolveMigrationsDir([fromArg], Env(("MIGRATIONS_DIR", fromEnv)), [_root]);

        Assert.Equal(fromArg, resolved);
    }

    [Fact]
    public void ExplicitDirThatDoesNotExist_Throws_InsteadOfProbing()
    {
        // A probe-able repo layout exists, but an explicit (wrong) path must never fall back to it.
        Directory.CreateDirectory(Path.Combine(_root, "services", "db", "migrations"));
        string missing = Path.Combine(_root, "nope");

        var ex = Assert.Throws<DirectoryNotFoundException>(() =>
            Migrator.ResolveMigrationsDir([], Env(("MIGRATIONS_DIR", missing)), [_root]));
        Assert.Contains(missing, ex.Message);
    }

    [Fact]
    public void WithoutExplicitDir_ProbesUpwardForTheRepoLayout()
    {
        string migrations = Directory.CreateDirectory(Path.Combine(_root, "services", "db", "migrations")).FullName;
        string deep = Directory.CreateDirectory(Path.Combine(_root, "a", "b", "c")).FullName;

        string resolved = Migrator.ResolveMigrationsDir([], Env(), [deep]);

        Assert.Equal(migrations, resolved);
    }

    [Fact]
    public void BlankEnv_IsTreatedAsUnset()
    {
        string migrations = Directory.CreateDirectory(Path.Combine(_root, "services", "db", "migrations")).FullName;

        string resolved = Migrator.ResolveMigrationsDir([], Env(("MIGRATIONS_DIR", "  ")), [_root]);

        Assert.Equal(migrations, resolved);
    }

    [Fact]
    public void NothingFound_Throws()
    {
        // A bare "migrations" folder is not the repo layout and must not be picked up.
        Directory.CreateDirectory(Path.Combine(_root, "migrations"));

        Assert.Throws<DirectoryNotFoundException>(() => Migrator.ResolveMigrationsDir([], Env(), [_root]));
    }

    [Fact]
    public async Task RunAsync_MissingMigrationsDir_ExitsNonZero()
    {
        var stdout = new StringWriter();
        var stderr = new StringWriter();

        int code = await Migrator.RunAsync(
            [], Env(("MIGRATIONS_DIR", Path.Combine(_root, "nope"))), stdout, stderr);

        Assert.Equal(Migrator.Failure, code);
        Assert.Contains("DirectoryNotFoundException", stderr.ToString());
    }

    [Fact]
    public async Task RunAsync_UnreachableDatabase_ExitsNonZero()
    {
        File.WriteAllText(Path.Combine(_root, "0001_x.sql"), "SELECT 1;");
        var stderr = new StringWriter();

        // "invalid." is reserved (RFC 6761) and never resolves: fails fast and deterministically,
        // with no Postgres, Docker or TCP connect involved.
        int code = await Migrator.RunAsync(
            [_root],
            Env(("SOLIA_DB_CONNECTION", "Host=invalid.;Timeout=3;Username=u;Password=p;Database=d")),
            new StringWriter(),
            stderr);

        Assert.Equal(Migrator.Failure, code);
        Assert.Contains("[migrator] FAILED", stderr.ToString());
    }

    [Fact]
    public void LockTimeout_DefaultsWhenUnsetOrBlank()
    {
        Assert.Equal(Migrator.DefaultLockTimeout, Migrator.ResolveLockTimeout(Env()));
        Assert.Equal(Migrator.DefaultLockTimeout, Migrator.ResolveLockTimeout(Env(("MIGRATOR_LOCK_TIMEOUT_SECONDS", " "))));
    }

    [Theory]
    [InlineData("0", 0)]
    [InlineData("45", 45)]
    public void LockTimeout_ReadsWholeSeconds(string raw, int seconds) =>
        Assert.Equal(
            TimeSpan.FromSeconds(seconds), Migrator.ResolveLockTimeout(Env(("MIGRATOR_LOCK_TIMEOUT_SECONDS", raw))));

    [Theory]
    [InlineData("-1")]
    [InlineData("1.5")]
    [InlineData("ten")]
    public void LockTimeout_Invalid_Throws(string raw)
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            Migrator.ResolveLockTimeout(Env(("MIGRATOR_LOCK_TIMEOUT_SECONDS", raw))));
        Assert.Contains(raw, ex.Message);
    }

    [Fact]
    public async Task RunAsync_InvalidLockTimeout_ExitsNonZero_BeforeConnecting()
    {
        var stderr = new StringWriter();

        int code = await Migrator.RunAsync(
            [_root],
            Env(("SOLIA_DB_CONNECTION", "Host=invalid.;Username=u;Password=p;Database=d"),
                ("MIGRATOR_LOCK_TIMEOUT_SECONDS", "soon")),
            new StringWriter(),
            stderr);

        Assert.Equal(Migrator.Failure, code);
        Assert.Contains("MIGRATOR_LOCK_TIMEOUT_SECONDS", stderr.ToString());
    }
}
