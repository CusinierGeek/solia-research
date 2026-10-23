using System.Data.Common;
using Microsoft.EntityFrameworkCore.Diagnostics;

namespace Solia.Core.Data;

/// <summary>
/// Rejects every command that is not a <c>SELECT</c>, so the paths that bypass
/// <see cref="SoliaDbContext.SaveChanges(bool)"/> (<c>ExecuteUpdate</c>, <c>ExecuteDelete</c>,
/// <c>Database.ExecuteSql</c>) cannot write either. Stateless: one shared instance.
/// </summary>
internal sealed class ReadOnlyCommandInterceptor : DbCommandInterceptor
{
    public static readonly ReadOnlyCommandInterceptor Instance = new();

    private ReadOnlyCommandInterceptor()
    {
    }

    public override InterceptionResult<DbDataReader> ReaderExecuting(
        DbCommand command, CommandEventData eventData, InterceptionResult<DbDataReader> result)
    {
        EnsureSelect(command);
        return result;
    }

    public override ValueTask<InterceptionResult<DbDataReader>> ReaderExecutingAsync(
        DbCommand command, CommandEventData eventData, InterceptionResult<DbDataReader> result, CancellationToken cancellationToken = default)
    {
        EnsureSelect(command);
        return ValueTask.FromResult(result);
    }

    public override InterceptionResult<object> ScalarExecuting(
        DbCommand command, CommandEventData eventData, InterceptionResult<object> result)
    {
        EnsureSelect(command);
        return result;
    }

    public override ValueTask<InterceptionResult<object>> ScalarExecutingAsync(
        DbCommand command, CommandEventData eventData, InterceptionResult<object> result, CancellationToken cancellationToken = default)
    {
        EnsureSelect(command);
        return ValueTask.FromResult(result);
    }

    // A non-query is never a read.
    public override InterceptionResult<int> NonQueryExecuting(
        DbCommand command, CommandEventData eventData, InterceptionResult<int> result) =>
        throw Rejected(command);

    public override ValueTask<InterceptionResult<int>> NonQueryExecutingAsync(
        DbCommand command, CommandEventData eventData, InterceptionResult<int> result, CancellationToken cancellationToken = default) =>
        throw Rejected(command);

    private static void EnsureSelect(DbCommand command)
    {
        if (!command.CommandText.TrimStart().StartsWith("SELECT", StringComparison.OrdinalIgnoreCase))
        {
            throw Rejected(command);
        }
    }

    private static InvalidOperationException Rejected(DbCommand command) =>
        new($"SoliaDbContext is read-only; write through the raw-SQL stores. Rejected: {FirstLine(command.CommandText)}");

    private static string FirstLine(string sql)
    {
        string trimmed = sql.TrimStart();
        int end = trimmed.IndexOfAny(['\r', '\n']);
        return end < 0 ? trimmed : trimmed[..end];
    }
}
