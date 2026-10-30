namespace Solia.DataPipeline.Retention;

/// <summary>
/// The only tables the purge touches, each aged by its <c>ts</c> column. <c>trades</c>, <c>models</c>,
/// <c>training_events</c>, <c>tokens_history</c> and <c>tokens_live</c> are deliberately absent: P&amp;L and model
/// history stay complete (the FKs into the purged tables are <c>ON DELETE SET NULL</c>).
/// </summary>
public enum RetainedTable
{
    // Purge order: predictions before the features they reference.
    Predictions,
    Features,
    Logs,
}

public static class RetainedTables
{
    public static IReadOnlyList<RetainedTable> All { get; } = Enum.GetValues<RetainedTable>();

    /// <summary>The SQL table name: a closed set, so it is safe to splice into a statement.</summary>
    public static string SqlName(this RetainedTable table) => table switch
    {
        RetainedTable.Predictions => "predictions",
        RetainedTable.Features => "features",
        RetainedTable.Logs => "logs",
        _ => throw new ArgumentOutOfRangeException(nameof(table), table, null),
    };
}
