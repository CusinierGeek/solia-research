using Npgsql;
using Solia.Shared.Configuration;

namespace Solia.Shared.Tests;

public class DbConnectionSettingsTests
{
    private static Func<string, string?> Lookup(params (string Key, string Value)[] values)
    {
        var map = values.ToDictionary(v => v.Key, v => v.Value);
        return key => map.GetValueOrDefault(key);
    }

    [Fact]
    public void Defaults_target_local_compose_postgres()
    {
        var cs = new NpgsqlConnectionStringBuilder(DbConnectionSettings.Resolve(Lookup()));

        Assert.Equal("localhost", cs.Host);
        Assert.Equal(5432, cs.Port);
        Assert.Equal("solia", cs.Database);
        Assert.Equal("solia", cs.Username);
        Assert.Equal("change_me", cs.Password);
    }

    [Fact]
    public void Explicit_connection_string_wins_over_parts()
    {
        string resolved = DbConnectionSettings.Resolve(Lookup(
            ("SOLIA_DB_CONNECTION", "Host=db;Database=x"),
            ("POSTGRES_HOST", "ignored")));

        Assert.Equal("Host=db;Database=x", resolved);
    }

    [Fact]
    public void Parts_override_defaults_and_special_characters_survive()
    {
        var cs = new NpgsqlConnectionStringBuilder(DbConnectionSettings.Resolve(Lookup(
            ("POSTGRES_HOST", "postgres"),
            ("POSTGRES_PORT", "6543"),
            ("POSTGRES_PASSWORD", "ab;cd='e\""))));

        Assert.Equal("postgres", cs.Host);
        Assert.Equal(6543, cs.Port);
        Assert.Equal("ab;cd='e\"", cs.Password);
    }

    [Fact]
    public void Trailing_carriage_return_from_windows_env_file_is_trimmed()
    {
        var cs = new NpgsqlConnectionStringBuilder(DbConnectionSettings.Resolve(Lookup(
            ("POSTGRES_PORT", "5433\r"),
            ("POSTGRES_HOST", "localhost\r"))));

        Assert.Equal(5433, cs.Port);
        Assert.Equal("localhost", cs.Host);
    }

    [Theory]
    [InlineData("tcp://10.0.0.5:5432")]
    [InlineData("0")]
    [InlineData("70000")]
    public void Invalid_port_fails_with_the_variable_name(string port)
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => DbConnectionSettings.Resolve(Lookup(("POSTGRES_PORT", port))));

        Assert.Contains("POSTGRES_PORT", ex.Message);
    }
}
