using Npgsql;
using NpgsqlTypes;

namespace Solia.Shared.Data;

/// <summary>
/// Helpers for building <see cref="NpgsqlParameter"/>s, shared by the Postgres sinks/stores so the
/// name/type/value mapping lives in one place. Import with <c>using static</c> to keep call sites terse.
/// </summary>
public static class NpgsqlParameters
{
    /// <summary>
    /// A named, explicitly-typed parameter. A null <paramref name="value"/> becomes a typed SQL NULL
    /// (<see cref="DBNull.Value"/>) rather than tripping Npgsql's type inference.
    /// </summary>
    public static NpgsqlParameter Param(string name, NpgsqlDbType type, object? value) =>
        new(name, type) { Value = value ?? DBNull.Value };
}
