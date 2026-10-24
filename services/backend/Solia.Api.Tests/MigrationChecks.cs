using System.Text.RegularExpressions;

namespace Solia.Api.Tests;

/// <summary>Reads CHECK constraints from services/db/migrations, the schema's source of truth.</summary>
internal static class MigrationChecks
{
    /// <summary>The values of the last <c>column IN (...)</c> CHECK across the migrations, in order.</summary>
    public static string[] InList(string column)
    {
        string? check = null;
        foreach (string file in Directory.GetFiles(PostgresFixture.FindMigrationsDir(), "*.sql").Order(StringComparer.Ordinal))
        {
            foreach (Match match in Regex.Matches(File.ReadAllText(file), $@"\b{column}\s+IN\s*\(([^)]*)\)", RegexOptions.IgnoreCase))
            {
                check = match.Groups[1].Value;
            }
        }

        Assert.NotNull(check);
        return [.. Regex.Matches(check, "'([^']*)'").Select(m => m.Groups[1].Value)];
    }
}
