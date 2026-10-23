using System.Diagnostics.CodeAnalysis;
using System.Security.Cryptography;

namespace Solia.Secrets;

/// <summary>An in-memory <see cref="ISecretsProvider"/> over an already-decrypted map.</summary>
public sealed class SecretsProvider : ISecretsProvider
{
    /// <summary>Config key overriding <see cref="SecretsFile.DefaultPath"/>.</summary>
    public const string FileKey = "SOLIA_SECRETS_FILE";

    private readonly Dictionary<string, string> _secrets;

    public SecretsProvider(IReadOnlyDictionary<string, string> secrets)
    {
        ArgumentNullException.ThrowIfNull(secrets);
        _secrets = new Dictionary<string, string>(secrets, StringComparer.Ordinal);
    }

    /// <summary>No secrets: every lookup falls back to plaintext config.</summary>
    public static SecretsProvider Empty { get; } = new(new Dictionary<string, string>());

    public IReadOnlyCollection<string> Names => _secrets.Keys;

    public bool TryGet(string name, [NotNullWhen(true)] out string? value) => _secrets.TryGetValue(name, out value);

    public string Get(string name) =>
        TryGet(name, out string? value) ? value : throw new SecretsException($"Secret '{name}' is not defined.");

    /// <summary>
    /// Decrypts the secrets file at startup. The path is <c>SOLIA_SECRETS_FILE</c>, else
    /// <see cref="SecretsFile.DefaultPath"/>; the key is <c>SOLIA_MASTER_KEY</c>.
    /// <list type="bullet">
    /// <item>No file at the default path → <see cref="Empty"/> (plaintext config is used, e.g. local dev without
    /// secrets).</item>
    /// <item>An explicit <c>SOLIA_SECRETS_FILE</c> that doesn't exist (typo, missing mount) →
    /// <see cref="SecretsException"/>: asking for a file and not getting it is a misconfiguration.</item>
    /// <item>A file but no key, a bad key, a wrong key or a tampered file → <see cref="SecretsException"/>:
    /// the host fails fast rather than silently running on plaintext defaults.</item>
    /// </list>
    /// </summary>
    /// <param name="lookup">Reads a flat key, e.g. <c>key =&gt; configuration[key]</c>.</param>
    public static SecretsProvider Load(Func<string, string?> lookup)
    {
        ArgumentNullException.ThrowIfNull(lookup);

        string? configuredPath = lookup(FileKey);
        bool explicitPath = !string.IsNullOrWhiteSpace(configuredPath);
        string path = explicitPath ? configuredPath!.Trim() : SecretsFile.DefaultPath;
        if (!File.Exists(path))
        {
            return explicitPath
                ? throw new SecretsException($"{FileKey} points to '{path}', which does not exist.")
                : Empty;
        }

        byte[] key = MasterKey.Parse(lookup(MasterKey.EnvironmentKey));
        try
        {
            return new SecretsProvider(SecretsFile.Read(path, key));
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
        }
    }
}
