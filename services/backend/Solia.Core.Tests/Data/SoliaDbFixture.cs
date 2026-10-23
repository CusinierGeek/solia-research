using Microsoft.EntityFrameworkCore;
using Solia.Core.Data;

namespace Solia.Core.Tests.Data;

/// <summary>
/// The shared <see cref="PostgresFixture"/> (Sprint-1 migrations applied), plus EF contexts over it, so the
/// model is exercised against the exact production schema (DDL = raw SQL, never EF-generated).
/// </summary>
public sealed class SoliaDbFixture : PostgresFixture
{
    /// <summary>A fresh context (caller disposes) with the same tracking default as the host wiring.</summary>
    public SoliaDbContext NewContext() =>
        new(new DbContextOptionsBuilder<SoliaDbContext>().UseNpgsql(ConnectionString).Options);
}
