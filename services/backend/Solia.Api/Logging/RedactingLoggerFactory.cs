using System.Collections;
using Microsoft.Extensions.Logging;
using Solia.Secrets;

namespace Solia.Api.Logging;

/// <summary>
/// Decorates the host's <see cref="ILoggerFactory"/> so every logger it hands out scrubs known secret values
/// (<see cref="SecretRedactor"/>) before any provider — console, the `logs`-table mirror, … — sees the entry.
/// Wrapping the factory rather than each provider keeps the per-provider filter rules (the <c>Postgres</c>
/// alias, …) intact. The container owns the inner factory, so disposing this one is a no-op.
/// </summary>
internal sealed class RedactingLoggerFactory(ILoggerFactory inner, SecretRedactor redactor) : ILoggerFactory
{
    public ILogger CreateLogger(string categoryName)
    {
        ILogger logger = inner.CreateLogger(categoryName);
        return redactor.IsEmpty ? logger : new RedactingLogger(logger, redactor);
    }

    public void AddProvider(ILoggerProvider provider) => inner.AddProvider(provider);

    public void Dispose()
    {
    }
}

/// <summary>
/// Passes entries through untouched unless the message, a structured value, the exception or a scope carries
/// a secret; then it forwards a redacted copy instead (message, structured values and exception text).
/// <para>The no-secret path is kept cheap: for structured state (the normal <c>LogX("… {Arg}", …)</c> case,
/// whose <c>{OriginalFormat}</c> is itself one of the values) only the values are scanned — the message is
/// formatted once, and only when something must be redacted — and exceptions are scanned by their message
/// chain, not their stack-trace-heavy <c>ToString()</c>.</para>
/// </summary>
internal sealed class RedactingLogger(ILogger inner, SecretRedactor redactor) : ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        List<KeyValuePair<string, object?>>? values = Values(state);
        string? text = values is null ? state.ToString() : null;
        if (values is null ? !redactor.ContainsSecret(text) : !values.Any(kv => ValueLeaks(kv.Value)))
        {
            return inner.BeginScope(state);
        }

        // A structured scope keeps its (redacted) values for scope-aware providers; its text is redacted too.
        return values is null
            ? inner.BeginScope(redactor.Redact(text!))
            : inner.BeginScope(new RedactedState(redactor.Redact(state.ToString() ?? string.Empty), Redact(values)));
    }

    public bool IsEnabled(LogLevel logLevel) => inner.IsEnabled(logLevel);

    public void Log<TState>(
        LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        if (!inner.IsEnabled(logLevel))
        {
            return;
        }

        ArgumentNullException.ThrowIfNull(formatter);
        List<KeyValuePair<string, object?>>? values = state is null ? null : Values(state);
        bool exceptionLeaks = exception is not null && ExceptionLeaks(exception);
        string? message = null;
        bool stateLeaks = values is null
            ? redactor.ContainsSecret(message = formatter(state, exception))
            : values.Any(kv => ValueLeaks(kv.Value));
        if (!exceptionLeaks && !stateLeaks)
        {
            inner.Log(logLevel, eventId, state, exception, formatter);
            return;
        }

        message ??= formatter(state, exception);
        var redacted = new RedactedState(redactor.Redact(message), values is null ? [] : Redact(values));
        Exception? safeException = exceptionLeaks ? new RedactedException(exception!, redactor) : exception;
        inner.Log(logLevel, eventId, redacted, safeException, static (s, _) => s.Message);
    }

    /// <summary>
    /// The structured values of a state or scope: the logging abstractions' key/value list, or any dictionary
    /// (e.g. <c>Dictionary&lt;string, string&gt;</c> scopes). Null for anything else (a plain string, …).
    /// </summary>
    private static List<KeyValuePair<string, object?>>? Values(object state) => state switch
    {
        string => null,
        IEnumerable<KeyValuePair<string, object?>> pairs => pairs.ToList(),
        IDictionary dictionary => Entries(dictionary),
        _ => null,
    };

    // The dictionary's own enumerator: a generic Dictionary's IEnumerable yields KeyValuePairs, not entries.
    private static List<KeyValuePair<string, object?>> Entries(IDictionary dictionary)
    {
        var values = new List<KeyValuePair<string, object?>>(dictionary.Count);
        IDictionaryEnumerator entries = dictionary.GetEnumerator();
        while (entries.MoveNext())
        {
            values.Add(new KeyValuePair<string, object?>(entries.Key.ToString() ?? string.Empty, entries.Value));
        }
        return values;
    }

    /// <summary>A value leaks if its text does — or, for a collection (rendered item by item), any item's.</summary>
    private bool ValueLeaks(object? value) => value switch
    {
        null => false,
        string text => redactor.ContainsSecret(text),
        IEnumerable items => items.Cast<object?>().Any(item => redactor.ContainsSecret(item?.ToString())),
        _ => redactor.ContainsSecret(value.ToString()),
    };

    private string? RedactValue(object? value) => value switch
    {
        null => null,
        string text => redactor.Redact(text),
        IEnumerable items => redactor.Redact(string.Join(", ", items.Cast<object?>().Select(item => item?.ToString()))),
        _ => redactor.Redact(value.ToString() ?? string.Empty),
    };

    private bool ExceptionLeaks(Exception exception)
    {
        var pending = new Stack<Exception>([exception]);
        while (pending.TryPop(out Exception? current))
        {
            if (redactor.ContainsSecret(current.Message))
            {
                return true;
            }
            if (current is AggregateException aggregate)
            {
                aggregate.InnerExceptions.ToList().ForEach(pending.Push);
            }
            else if (current.InnerException is not null)
            {
                pending.Push(current.InnerException);
            }
        }
        return false;
    }

    private List<KeyValuePair<string, object?>> Redact(List<KeyValuePair<string, object?>> values) =>
        values.Select(kv => new KeyValuePair<string, object?>(kv.Key, RedactValue(kv.Value))).ToList();

    /// <summary>The scrubbed entry: the redacted message plus redacted structured values (for structured sinks).</summary>
    private sealed class RedactedState(string message, List<KeyValuePair<string, object?>> values)
        : IReadOnlyList<KeyValuePair<string, object?>>
    {
        public string Message { get; } = message;

        public KeyValuePair<string, object?> this[int index] => values[index];

        public int Count => values.Count;

        public IEnumerator<KeyValuePair<string, object?>> GetEnumerator() => values.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString() => Message;
    }

    /// <summary>Stands in for an exception whose text carries a secret: same type name, scrubbed text.</summary>
    private sealed class RedactedException(Exception original, SecretRedactor redactor)
        : Exception(redactor.Redact(original.Message))
    {
        private readonly string _text = redactor.Redact(original.ToString());

        public override string? StackTrace => null;

        public override string ToString() => _text;
    }
}
