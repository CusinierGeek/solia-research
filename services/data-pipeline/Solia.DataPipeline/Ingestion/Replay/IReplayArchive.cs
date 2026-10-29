namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Where a replay reads its hourly archives from: the HTTP source directly
/// (<see cref="HttpReplayArchiveOpener"/>) or a local <see cref="ArchiveMirror"/> in front of it.
/// </summary>
public interface IReplayArchive
{
    /// <summary>The compressed archive for <paramref name="hour"/>, or null when the source has no such hour.</summary>
    Task<Stream?> OpenAsync(DateTimeOffset hour, CancellationToken cancellationToken);
}
