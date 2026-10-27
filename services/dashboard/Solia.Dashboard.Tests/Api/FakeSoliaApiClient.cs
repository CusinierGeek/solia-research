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

    public Func<PredictionHistoryQuery, CancellationToken, Task<PagedResult<PredictionRecordDto>>> PredictionHistory { get; set; } =
        (q, _) => Task.FromResult(new PagedResult<PredictionRecordDto>([], q.Page ?? 1, q.PageSize ?? 50, 0));

    public List<PredictionHistoryQuery> PredictionHistoryCalls { get; } = [];

    public Task<PagedResult<PredictionRecordDto>> GetPredictionHistoryAsync(PredictionHistoryQuery? query = null, CancellationToken cancellationToken = default)
    {
        query ??= new();
        PredictionHistoryCalls.Add(query);
        return PredictionHistory(query, cancellationToken);
    }

    public Task<IReadOnlyList<LiveTokenDto>> GetLiveTokensAsync(LiveTokensQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<PagedResult<TokenHistoryDto>> GetTokenHistoryAsync(TokenHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Func<TradeHistoryQuery, CancellationToken, Task<PagedResult<TradeDto>>> TradeHistory { get; set; } =
        (q, _) => Task.FromResult(new PagedResult<TradeDto>([], q.Page ?? 1, q.PageSize ?? 50, 0));

    public List<TradeHistoryQuery> TradeHistoryCalls { get; } = [];

    public Task<PagedResult<TradeDto>> GetTradeHistoryAsync(TradeHistoryQuery? query = null, CancellationToken cancellationToken = default)
    {
        query ??= new();
        TradeHistoryCalls.Add(query);
        return TradeHistory(query, cancellationToken);
    }

    public Task<WinRateResponse> GetWinRateAsync(StatsWindow? window = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<PerformanceResponse> GetPerformanceAsync(int? limit = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<IReadOnlyList<LogDto>> GetBotLogsAsync(BotLogsQuery? query = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<ModelStatusResponse> GetModelStatusAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}
