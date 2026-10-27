using System.Text.Json;
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

    public Func<StatsWindow?, CancellationToken, Task<WinRateResponse>> WinRate { get; set; } =
        (w, _) => Task.FromResult(new WinRateResponse(0, 0, 0, 0, w is { } v ? QueryValues.Format(v) : "all"));

    public List<StatsWindow?> WinRateCalls { get; } = [];

    public Task<WinRateResponse> GetWinRateAsync(StatsWindow? window = null, CancellationToken cancellationToken = default)
    {
        WinRateCalls.Add(window);
        return WinRate(window, cancellationToken);
    }

    public Func<int?, CancellationToken, Task<PerformanceResponse>> Performance { get; set; } =
        (_, _) => Task.FromResult(new PerformanceResponse([], 0m));

    public List<int?> PerformanceCalls { get; } = [];

    public Task<PerformanceResponse> GetPerformanceAsync(int? limit = null, CancellationToken cancellationToken = default)
    {
        PerformanceCalls.Add(limit);
        return Performance(limit, cancellationToken);
    }

    public Func<BotLogsQuery, CancellationToken, Task<IReadOnlyList<LogDto>>> BotLogs { get; set; } =
        (_, _) => Task.FromResult<IReadOnlyList<LogDto>>([]);

    public List<BotLogsQuery> BotLogsCalls { get; } = [];

    public Task<IReadOnlyList<LogDto>> GetBotLogsAsync(BotLogsQuery? query = null, CancellationToken cancellationToken = default)
    {
        query ??= new();
        BotLogsCalls.Add(query);
        return BotLogs(query, cancellationToken);
    }

    public Func<CancellationToken, Task<ModelStatusResponse>> ModelStatus { get; set; } =
        _ => Task.FromResult(new ModelStatusResponse(null, null, new Dictionary<string, JsonElement>(), 0, 500));

    public int ModelStatusCalls { get; private set; }

    public Task<ModelStatusResponse> GetModelStatusAsync(CancellationToken cancellationToken = default)
    {
        ModelStatusCalls++;
        return ModelStatus(cancellationToken);
    }
}
