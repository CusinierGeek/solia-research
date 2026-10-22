using Npgsql;
using Testcontainers.PostgreSql;

namespace Solia.DataPipeline.Tests.Pipeline;

/// <summary>An ephemeral PostgreSQL 16 with the T1 migrations applied, shared by a test class.</summary>
public sealed class PostgresFixture : IAsyncLifetime
{
#pragma warning disable CS0618 // builder ctor is obsolete; .WithImage still pins the image
    private readonly PostgreSqlContainer _postgres =
        new PostgreSqlBuilder().WithImage("postgres:16-alpine").Build();
#pragma warning restore CS0618

    public NpgsqlDataSource DataSource { get; private set; } = null!;

    public async Task InitializeAsync()
    {
        await _postgres.StartAsync();
        DataSource = NpgsqlDataSource.Create(_postgres.GetConnectionString());
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
