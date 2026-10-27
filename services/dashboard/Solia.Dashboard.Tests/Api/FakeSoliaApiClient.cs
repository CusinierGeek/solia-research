using Solia.Dashboard.Api;

namespace Solia.Dashboard.Tests.Api;

/// <summary>
/// An <see cref="ISoliaApiClient"/> for component tests: each endpoint a view uses answers through a settable
/// delegate and records its queries; the others are not expected to be called.
/// </summary>
internal sealed class FakeSoliaApiClient : ISoliaApiClient
{
    public Func<LivePredictionsQuery, CancellationToken, Task<IReadOnlyList<PredictionRecordDto>>> LivePredictions { get; set; } =
        (_, _) => Task.FromResult<IReadOnlyList<PredictionRecordDto>>([]);

    public List<LivePredictionsQuery> LivePredictionsCalls { get; } = [];

    public Task<IReadOnlyList<PredictionRecordDto>> GetLivePredictionsAsync(LivePredictionsQuery? query = null, CancellationToken cancellationToken = default)
    {
        query ??= new();
        LivePredictionsCalls.Add(query);
        return LivePredictions(query, cancellationToken);
    }

    public Task<IReadOnlyList<LiveTokenDto>> GetLiveTokensAsync(LiveTokensQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<PagedResult<TokenHistoryDto>> GetTokenHistoryAsync(TokenHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<PagedResult<PredictionRecordDto>> GetPredictionHistoryAsync(PredictionHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<PagedResult<TradeDto>> GetTradeHistoryAsync(TradeHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<WinRateResponse> GetWinRateAsync(StatsWindow? window = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<PerformanceResponse> GetPerformanceAsync(int? limit = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<IReadOnlyList<LogDto>> GetBotLogsAsync(BotLogsQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<ModelStatusResponse> GetModelStatusAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}
