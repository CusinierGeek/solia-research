using System.Diagnostics.CodeAnalysis;

namespace Solia.Secrets;

/// <summary>
/// Read access to the decrypted secrets, held in memory only (see docs/SECRETS.md). Never log what it returns.
/// </summary>
public interface ISecretsProvider
{
    /// <summary>The names of the secrets on hand (names are not sensitive; values are).</summary>
    IReadOnlyCollection<string> Names { get; }

    bool TryGet(string name, [NotNullWhen(true)] out string? value);

    /// <summary>The secret's value; throws <see cref="SecretsException"/> (naming the secret) when it is absent.</summary>
    string Get(string name);
}

/// <summary>Well-known secret names in <c>secrets.enc</c>.</summary>
public static class SecretNames
{
    /// <summary>A full Postgres connection string; wins over everything else.</summary>
    public const string DbConnection = "db_connection";

    /// <summary>The Postgres password, overlaid on the configured connection string.</summary>
    public const string DbPassword = "db_password";
}
