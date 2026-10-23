using System.Security.Cryptography;

namespace Solia.Secrets;

/// <summary>The AES-256 master key: 32 bytes, carried base64-encoded in <c>SOLIA_MASTER_KEY</c>.</summary>
public static class MasterKey
{
    /// <summary>The env var / config key holding the base64 master key.</summary>
    public const string EnvironmentKey = "SOLIA_MASTER_KEY";

    public const int SizeInBytes = 32;

    /// <summary>Decodes a base64 master key; throws <see cref="SecretsException"/> unless it is exactly 32 bytes.</summary>
    public static byte[] Parse(string? base64, string source = EnvironmentKey)
    {
        if (string.IsNullOrWhiteSpace(base64))
        {
            throw new SecretsException($"{source} is not set.");
        }

        byte[] key;
        try
        {
            key = Convert.FromBase64String(base64.Trim());
        }
        catch (FormatException ex)
        {
            throw new SecretsException($"{source} is not valid base64.", ex);
        }

        if (key.Length != SizeInBytes)
        {
            CryptographicOperations.ZeroMemory(key);
            throw new SecretsException($"{source} must decode to {SizeInBytes} bytes.");
        }
        return key;
    }

    /// <summary>A fresh random key, base64-encoded.</summary>
    public static string Generate() => Convert.ToBase64String(RandomNumberGenerator.GetBytes(SizeInBytes));
}
