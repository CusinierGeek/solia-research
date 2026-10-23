using System.Text.Json;
using Solia.Api.Logging;
using Solia.Bot.Logging;
using Solia.Secrets;
using Solia.Shared.Logging;

namespace Solia.Api.Tests;

/// <summary>The bot's `logs` rows bypass ILogger, so its sink scrubs them itself.</summary>
public class RedactingBotLogSinkTests
{
    private static readonly string Secret = FakeSecret.New();

    private readonly CapturingSink _inner = new();
    private readonly RedactingBotLogSink _sink;

    public RedactingBotLogSinkTests() => _sink = new RedactingBotLogSink(_inner, new SecretRedactor([Secret]));

    [Fact]
    public async Task A_secret_in_the_message_or_context_never_reaches_the_table()
    {
        await _sink.WriteAsync(LogSeverity.Error, $"bot failed: {Secret}",
            new { Error = $"connect to Password={Secret}", Attempts = 3, Hosts = new[] { "db", Secret } });

        (LogSeverity level, string message, string? context) = Assert.Single(_inner.Rows);
        Assert.Equal(LogSeverity.Error, level);
        Assert.Equal("bot failed: ***", message);
        Assert.DoesNotContain(Secret, context, StringComparison.Ordinal);

        using JsonDocument json = JsonDocument.Parse(context!);
        Assert.Equal("connect to Password=***", json.RootElement.GetProperty("error").GetString());
        Assert.Equal(3, json.RootElement.GetProperty("attempts").GetInt32());
        Assert.Equal("***", json.RootElement.GetProperty("hosts")[1].GetString());
    }

    [Fact]
    public async Task A_secret_nested_in_the_context_is_scrubbed()
    {
        await _sink.WriteAsync(LogSeverity.Warn, "retry", new { Inner = new { Detail = Secret } });

        Assert.Equal("""{"inner":{"detail":"***"}}""", Assert.Single(_inner.Rows).Context);
    }

    [Fact]
    public async Task A_clean_entry_is_forwarded_unchanged()
    {
        var context = new { Mint = "abc" };

        await _sink.WriteAsync(LogSeverity.Info, "position opened", context);

        Assert.Same(context, _inner.Contexts.Single());
        Assert.Equal("position opened", _inner.Rows.Single().Message);
    }

    [Fact]
    public async Task Without_secrets_nothing_is_reserialised()
    {
        var sink = new RedactingBotLogSink(_inner, new SecretRedactor([]));
        var context = new { Error = "x" };

        await sink.WriteAsync(LogSeverity.Info, "bot started", context);

        Assert.Same(context, _inner.Contexts.Single());
    }

    private sealed class CapturingSink : IBotLogSink
    {
        public List<(LogSeverity Level, string Message, string? Context)> Rows { get; } = [];

        public List<object?> Contexts { get; } = [];

        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
        {
            Contexts.Add(context);
            Rows.Add((level, message, PostgresLogSink.SerializeContext(context)));
            return Task.CompletedTask;
        }
    }
}
