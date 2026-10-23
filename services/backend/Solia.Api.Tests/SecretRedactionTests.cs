using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Solia.Api.Logging;
using Solia.Secrets;

namespace Solia.Api.Tests;

/// <summary>The logging pipeline never hands a known secret value to a provider.</summary>
public sealed class SecretRedactionTests : IDisposable
{
    private static readonly string Secret = FakeSecret.New();

    private readonly CapturingProvider _captured = new();
    private readonly ServiceProvider _services;
    private readonly ILogger _logger;

    public SecretRedactionTests()
    {
        _services = new ServiceCollection()
            .AddSingleton(new SecretRedactor([Secret]))
            .AddLogging(logging => logging
                .SetMinimumLevel(LogLevel.Trace)
                .AddProvider(_captured)
                .AddSecretRedaction())
            .BuildServiceProvider();
        _logger = _services.GetRequiredService<ILoggerFactory>().CreateLogger("backend");
    }

    public void Dispose() => _services.Dispose();

    [Fact]
    public void A_logged_line_containing_a_secret_is_scrubbed()
    {
        _logger.LogWarning("Connecting with password {Password} to {Host}", Secret, "db");

        Entry entry = Assert.Single(_captured.Entries);
        Assert.Equal("Connecting with password *** to db", entry.Message);
        Assert.DoesNotContain(Secret, entry.Message, StringComparison.Ordinal);
        Assert.Equal("***", entry.Values["Password"]);
        Assert.Equal("db", entry.Values["Host"]);
    }

    [Fact]
    public void A_secret_inside_an_exception_is_scrubbed()
    {
        var failure = new InvalidOperationException($"auth failed for {Secret}");

        _logger.LogError(failure, "Database unreachable");

        Entry entry = Assert.Single(_captured.Entries);
        Assert.Equal("Database unreachable", entry.Message);
        Assert.Contains("InvalidOperationException", entry.Exception!.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain(Secret, entry.Exception.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain(Secret, entry.Exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_secret_in_a_scope_is_scrubbed()
    {
        using (_logger.BeginScope("token={Token}", Secret))
        {
            _logger.LogInformation("inside");
        }

        Assert.Equal(["token=***"], _captured.Scopes);
    }

    [Fact]
    public void A_secret_in_a_dictionary_scope_is_scrubbed_value_by_value()
    {
        using (_logger.BeginScope(new Dictionary<string, object> { ["conn"] = Secret, ["host"] = "db" }))
        using (_logger.BeginScope(new Dictionary<string, string> { ["token"] = Secret }))
        {
            _logger.LogInformation("inside");
        }

        Assert.Equal(2, _captured.ScopeValues.Count);
        Assert.Equal("***", _captured.ScopeValues[0]["conn"]);
        Assert.Equal("db", _captured.ScopeValues[0]["host"]);
        Assert.Equal("***", _captured.ScopeValues[1]["token"]);
        Assert.DoesNotContain(_captured.Scopes, s => s.Contains(Secret, StringComparison.Ordinal));
    }

    [Fact]
    public void A_scope_without_secrets_is_forwarded_as_is()
    {
        var scope = new Dictionary<string, object> { ["host"] = "db" };

        using (_logger.BeginScope(scope))
        {
            _logger.LogInformation("inside");
        }

        Assert.Same(scope, _captured.ScopeStates.Single());
    }

    [Fact]
    public void A_secret_inside_a_collection_argument_is_scrubbed()
    {
        _logger.LogWarning("Tried {Hosts}", new List<string> { "db", Secret });

        Entry entry = Assert.Single(_captured.Entries);
        Assert.DoesNotContain(Secret, entry.Message, StringComparison.Ordinal);
        Assert.Equal("db, ***", entry.Values["Hosts"]);
    }

    [Fact]
    public void A_secret_in_an_inner_or_aggregated_exception_is_scrubbed()
    {
        var failure = new AggregateException(
            new InvalidOperationException("outer", new TimeoutException($"inner {Secret}")));

        _logger.LogError(failure, "Batch failed");

        Entry entry = Assert.Single(_captured.Entries);
        Assert.NotSame(failure, entry.Exception);
        Assert.DoesNotContain(Secret, entry.Exception!.ToString(), StringComparison.Ordinal);
        Assert.Contains("TimeoutException", entry.Exception.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void An_interpolated_secret_in_a_plain_message_is_scrubbed()
    {
#pragma warning disable CA2254 // The point: a caller that interpolates instead of using a template.
        _logger.LogInformation($"token is {Secret}");
#pragma warning restore CA2254

        Entry entry = Assert.Single(_captured.Entries);
        Assert.Equal("token is ***", entry.Message);
        Assert.Equal("token is ***", entry.Values["{OriginalFormat}"]);
    }

    [Fact]
    public void A_line_without_secrets_passes_through_untouched()
    {
        var failure = new InvalidOperationException("plain failure");

        _logger.LogError(failure, "Nothing to hide in {Thing}", "here");

        Entry entry = Assert.Single(_captured.Entries);
        Assert.Equal("Nothing to hide in here", entry.Message);
        Assert.Same(failure, entry.Exception);
        Assert.Equal("Nothing to hide in {Thing}", entry.Values["{OriginalFormat}"]);
    }

    [Fact]
    public void No_secrets_means_the_stock_loggers_are_used()
    {
        using ServiceProvider services = new ServiceCollection()
            .AddSingleton(new SecretRedactor([]))
            .AddLogging(logging => logging.AddSecretRedaction())
            .BuildServiceProvider();

        ILogger logger = services.GetRequiredService<ILoggerFactory>().CreateLogger("backend");

        Assert.IsNotType<RedactingLogger>(logger);
    }

    private sealed record Entry(string Message, Exception? Exception, Dictionary<string, object?> Values);

    private sealed class CapturingProvider : ILoggerProvider
    {
        public List<Entry> Entries { get; } = [];

        public List<string> Scopes { get; } = [];

        public List<object> ScopeStates { get; } = [];

        public List<Dictionary<string, object?>> ScopeValues { get; } = [];

        public ILogger CreateLogger(string categoryName) => new CapturingLogger(this);

        public void Dispose()
        {
        }

        private sealed class CapturingLogger(CapturingProvider owner) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull
            {
                owner.Scopes.Add(state.ToString()!);
                owner.ScopeStates.Add(state);
                if (state is IEnumerable<KeyValuePair<string, object?>> pairs)
                {
                    owner.ScopeValues.Add(pairs.ToDictionary(kv => kv.Key, kv => kv.Value));
                }
                return null;
            }

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(
                LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                var values = state is IEnumerable<KeyValuePair<string, object?>> pairs
                    ? pairs.ToDictionary(kv => kv.Key, kv => kv.Value)
                    : [];
                owner.Entries.Add(new Entry(formatter(state, exception), exception, values));
            }
        }
    }
}
