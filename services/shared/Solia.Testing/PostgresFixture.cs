using Npgsql;
using Testcontainers.PostgreSql;
using Xunit;

namespace Solia.Testing;

/// <summary>
/// An ephemeral PostgreSQL 16 with the <c>services/db/migrations</c> SQL applied, shared by a test class.
/// The single Testcontainers bootstrap for every .NET test suite, so the schema they run against can't drift.
/// Classes using it are integration tests: tag them <c>[Trait(TestCategories.Key, TestCategories.Integration)]</c>.
/// </summary>
public class PostgresFixture : IAsyncLifetime
{
#pragma warning disable CS0618 // builder ctor is obsolete; .WithImage still pins the image
    private readonly PostgreSqlContainer _postgres =
        new PostgreSqlBuilder().WithImage("postgres:16-alpine").Build();
#pragma warning restore CS0618

    public string ConnectionString { get; private set; } = null!;

    public NpgsqlDataSource DataSource { get; private set; } = null!;

    public async Task InitializeAsync()
    {
        await _postgres.StartAsync();
        ConnectionString = _postgres.GetConnectionString();
        DataSource = NpgsqlDataSource.Create(ConnectionString);
        foreach (string file in Directory.GetFiles(FindMigrationsDir(), "*.sql").OrderBy(f => f, StringComparer.Ordinal))
        {
            await using NpgsqlCommand cmd = DataSource.CreateCommand(await File.ReadAllTextAsync(file));
            await cmd.ExecuteNonQueryAsync();
        }
    }

    public async Task DisposeAsync()
    {
        await DataSource.DisposeAsync();
        await _postgres.DisposeAsync();
    }

    public async Task<T> ScalarAsync<T>(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        return (T)Convert.ChangeType((await cmd.ExecuteScalarAsync())!, typeof(T));
    }

    private static string FindMigrationsDir()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "services", "db", "migrations");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate services/db/migrations from the test output.");
    }
}
