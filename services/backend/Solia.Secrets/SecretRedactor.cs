using System.Buffers;

namespace Solia.Secrets;

/// <summary>
/// Scrubs known secret values out of text before it is emitted (logs). Every non-empty value is replaced
/// with <see cref="Mask"/>, longest first so a secret containing another is masked whole. Detection is a
/// single multi-string search (<see cref="SearchValues{T}"/>), so the no-secret path stays cheap however
/// many secrets there are.
/// </summary>
public sealed class SecretRedactor
{
    public const string Mask = "***";

    private readonly string[] _values;
    private readonly SearchValues<string>? _search;

    public SecretRedactor(IEnumerable<string?> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        _values = values
            .Where(v => !string.IsNullOrEmpty(v))
            .Select(v => v!)
            .Distinct(StringComparer.Ordinal)
            .OrderByDescending(v => v.Length)
            .ToArray();
        _search = _values.Length == 0 ? null : SearchValues.Create(_values, StringComparison.Ordinal);
    }

    /// <summary>
    /// Redacts every secret of <paramref name="provider"/>, plus any extra values: the master key, and the
    /// parts of composite secrets (e.g. the password inside <see cref="SecretNames.DbConnection"/>).
    /// </summary>
    public static SecretRedactor For(ISecretsProvider provider, params IEnumerable<string?> extraValues)
    {
        ArgumentNullException.ThrowIfNull(provider);
        return new SecretRedactor(provider.Names.Select(provider.Get).Concat(extraValues));
    }

    /// <summary>True when there is nothing to redact (lets callers skip the work).</summary>
    public bool IsEmpty => _search is null;

    public bool ContainsSecret(string? text) =>
        _search is not null && !string.IsNullOrEmpty(text) && text.AsSpan().ContainsAny(_search);

    public string Redact(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!ContainsSecret(text))
        {
            return text;
        }
        foreach (string value in _values)
        {
            text = text.Replace(value, Mask, StringComparison.Ordinal);
        }
        return text;
    }
}
