using NpgsqlTypes;
using Solia.Shared.Data;

namespace Solia.Shared.Tests;

public class NpgsqlParametersTests
{
    [Fact]
    public void Param_carries_the_name_type_and_value()
    {
        var p = NpgsqlParameters.Param("mint", NpgsqlDbType.Text, "abc");

        Assert.Equal("mint", p.ParameterName);
        Assert.Equal(NpgsqlDbType.Text, p.NpgsqlDbType);
        Assert.Equal("abc", p.Value);
    }

    [Fact]
    public void A_null_value_becomes_a_typed_sql_null()
    {
        var p = NpgsqlParameters.Param("prediction", NpgsqlDbType.Bigint, null);

        // Null must reach Postgres as a typed NULL, not trip Npgsql's type inference.
        Assert.Equal(NpgsqlDbType.Bigint, p.NpgsqlDbType);
        Assert.Equal(DBNull.Value, p.Value);
    }

    [Fact]
    public void A_boxed_value_type_is_preserved()
    {
        var at = new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

        var p = NpgsqlParameters.Param("opened", NpgsqlDbType.TimestampTz, at);

        Assert.Equal(at, p.Value);
    }
}
