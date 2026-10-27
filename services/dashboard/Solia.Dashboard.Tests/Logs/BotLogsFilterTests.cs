using Solia.Dashboard.Api;
using Solia.Dashboard.Logs;

namespace Solia.Dashboard.Tests.Logs;

public class BotLogsFilterTests
{
    private const string Base = "http://localhost/logs";

    private static LogDto Log(string level, string service = "bot") =>
        new(1, DateTimeOffset.UnixEpoch, service, level, "message", null);

    [Fact]
    public void No_query_is_no_filter()
    {
        BotLogsFilter filter = BotLogsFilter.FromUri(Base);

        Assert.Equal(BotLogsFilter.None, filter);
        Assert.Equal(BotLogsFilter.DefaultLimit, filter.EffectiveLimit);
        Assert.Equal(new BotLogsQuery(), filter.ToQuery());
    }

    [Fact]
    public void Reads_the_api_parameters_case_insensitively()
    {
        BotLogsFilter filter = BotLogsFilter.FromUri($"{Base}?level=WARN&service=data-pipeline&limit=250");

        Assert.Equal(new BotLogsFilter(BotLogLevel.Warn, LogService.DataPipeline, 250), filter);
        Assert.Equal(new BotLogsQuery { Level = BotLogLevel.Warn, Service = LogService.DataPipeline, Limit = 250 }, filter.ToQuery());
    }

    [Theory]
    [InlineData("level=loud&service=nope&limit=abc")]
    [InlineData("limit=0")]
    [InlineData("limit=-5")]
    [InlineData("limit=100")] // the default is not spelled out
    public void Ignores_values_that_do_not_parse(string query) =>
        Assert.Equal(BotLogsFilter.None, BotLogsFilter.FromUri($"{Base}?{query}"));

    [Fact]
    public void Caps_the_limit_to_the_api_maximum() =>
        Assert.Equal(BotLogsFilter.MaxLimit, BotLogsFilter.FromUri($"{Base}?limit=50000").Limit);

    [Theory]
    [InlineData("99999999999")]
    [InlineData("0000000000000000002000")]
    public void Caps_a_limit_too_large_for_an_int(string limit) =>
        Assert.Equal(BotLogsFilter.MaxLimit, BotLogsFilter.FromUri($"{Base}?limit={limit}").Limit);

    [Fact]
    public void The_default_limit_is_held_as_none()
    {
        Assert.Equal(BotLogsFilter.None, new BotLogsFilter(Limit: BotLogsFilter.DefaultLimit));
        Assert.Equal(BotLogsFilter.None, BotLogsFilter.None with { Limit = BotLogsFilter.DefaultLimit });
        Assert.Null((new BotLogsFilter(Limit: 250) with { Limit = BotLogsFilter.DefaultLimit }).Limit);
        Assert.Equal(250, (BotLogsFilter.None with { Limit = 250 }).Limit);
    }

    [Fact]
    public void Writes_only_what_is_set_back_to_the_url()
    {
        Assert.All(BotLogsFilter.None.ToQueryParameters().Values, Assert.Null);
        Assert.All(new BotLogsFilter(Limit: BotLogsFilter.DefaultLimit).ToQueryParameters().Values, Assert.Null);

        IReadOnlyDictionary<string, object?> parameters = new BotLogsFilter(BotLogLevel.Error, LogService.AiService, 500).ToQueryParameters();
        Assert.Equal("error", parameters["level"]);
        Assert.Equal("ai-service", parameters["service"]);
        Assert.Equal("500", parameters["limit"]);
    }

    [Theory]
    [InlineData(null, "debug", true)]
    [InlineData(null, "error", true)]
    [InlineData(BotLogLevel.Debug, "debug", true)]
    [InlineData(BotLogLevel.Info, "debug", false)]
    [InlineData(BotLogLevel.Info, "info", true)]
    [InlineData(BotLogLevel.Info, "warn", true)]
    [InlineData(BotLogLevel.Warn, "info", false)]
    [InlineData(BotLogLevel.Warn, "warn", true)]
    [InlineData(BotLogLevel.Warn, "error", true)]
    [InlineData(BotLogLevel.Error, "warn", false)]
    [InlineData(BotLogLevel.Error, "error", true)]
    [InlineData(BotLogLevel.Debug, "fatal", false)] // off the scale: never meets a minimum
    public void A_level_is_a_minimum_severity(BotLogLevel? min, string level, bool matches) =>
        Assert.Equal(matches, new BotLogsFilter(Level: min).Matches(Log(level)));

    [Theory]
    [InlineData(null, "backend", true)]
    [InlineData(LogService.Backend, "backend", true)]
    [InlineData(LogService.Backend, "bot", false)]
    [InlineData(LogService.DataPipeline, "data-pipeline", true)]
    [InlineData(LogService.AiService, "ai-service", true)]
    public void A_service_is_matched_by_its_wire_name(LogService? service, string logService, bool matches) =>
        Assert.Equal(matches, new BotLogsFilter(Service: service).Matches(Log("info", logService)));

    [Fact]
    public void Both_filters_must_match()
    {
        var filter = new BotLogsFilter(BotLogLevel.Warn, LogService.Bot);

        Assert.True(filter.Matches(Log("error", "bot")));
        Assert.False(filter.Matches(Log("error", "backend")));
        Assert.False(filter.Matches(Log("info", "bot")));
    }
}
