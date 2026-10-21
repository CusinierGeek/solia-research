namespace Solia.DataPipeline.Ingestion;

/// <summary>
/// Provider-agnostic classification of an ingested event. Mapped from the data source's
/// native event type (e.g. pumpapi's <c>action</c>) by the parser. <see cref="Other"/> covers
/// recognized-but-uninteresting events we still want to pass through without special handling.
/// </summary>
public enum EventKind
{
    Create,
    Buy,
    Sell,
    Migrate,
    CreatePool,
    AddLiquidity,
    RemoveLiquidity,
    Transfer,
    Other,
}
