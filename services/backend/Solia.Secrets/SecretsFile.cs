using System.Security.Cryptography;
using System.Text.Json;

namespace Solia.Secrets;

/// <summary>
/// The secrets file: a flat JSON object of string values (<c>{ "db_password": "..." }</c>) sealed with
/// <see cref="SecretsCipher"/>. Decrypted content only ever lives in memory — it is never written to disk.
/// </summary>
public static class SecretsFile
{
    /// <summary>Default location, relative to the working directory.</summary>
    public const string DefaultPath = "secrets/secrets.enc";

    private static ReadOnlySpan<byte> Utf8Bom => [0xEF, 0xBB, 0xBF];

    public static string Seal(IReadOnlyDictionary<string, string> secrets, ReadOnlySpan<byte> key)
    {
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(secrets);
        try
        {
            return SecretsCipher.Encrypt(json, key);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(json);
        }
    }

    public static Dictionary<string, string> Open(string sealedContent, ReadOnlySpan<byte> key)
    {
        byte[] json = SecretsCipher.Decrypt(sealedContent, key);
        try
        {
            return ParseJson(json);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(json);
        }
    }

    public static Dictionary<string, string> Read(string path, ReadOnlySpan<byte> key)
    {
        if (!File.Exists(path))
        {
            throw new SecretsException($"Secrets file '{path}' does not exist.");
        }
        return Open(File.ReadAllText(path), key);
    }

    /// <summary>
    /// Writes atomically (temp file + rename), so an interrupted rotate never leaves a half-written file. Each
    /// write uses its own temp file (concurrent writers can't clobber each other's ciphertext; the last rename
    /// wins whole), and a failed write removes it rather than leaving it next to the real file.
    /// </summary>
    public static void Write(string path, IReadOnlyDictionary<string, string> secrets, ReadOnlySpan<byte> key)
    {
        string sealedContent = Seal(secrets, key);
        string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string temp = $"{path}.{Guid.NewGuid():N}.tmp";
        try
        {
            File.WriteAllText(temp, sealedContent);
            File.Move(temp, path, overwrite: true);
        }
        catch
        {
            TryDelete(temp);
            throw;
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            // Best effort: the original failure is what the caller needs to see.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    /// <summary>
    /// Parses a plaintext secrets JSON object; every value must be a string. A leading UTF-8 BOM (Notepad,
    /// PowerShell 5.1 <c>-Encoding utf8</c>) is skipped; anything after the object is rejected, so a pasted
    /// second object can't be silently dropped.
    /// </summary>
    public static Dictionary<string, string> ParseJson(ReadOnlySpan<byte> json)
    {
        if (json.StartsWith(Utf8Bom))
        {
            json = json[Utf8Bom.Length..];
        }

        var reader = new Utf8JsonReader(json);
        try
        {
            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            if (reader.Read())
            {
                throw new SecretsException("Secrets JSON has content after the object.");
            }
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw new SecretsException("Secrets JSON must be an object of string values.");
            }

            var secrets = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (JsonProperty property in document.RootElement.EnumerateObject())
            {
                if (property.Value.ValueKind != JsonValueKind.String)
                {
                    // Name the key, never the value.
                    throw new SecretsException($"Secret '{property.Name}' must be a JSON string.");
                }
                secrets[property.Name] = property.Value.GetString()!;
            }
            return secrets;
        }
        catch (JsonException ex)
        {
            // JsonException messages can quote the offending input; keep only the position.
            // LineNumber is 0-based; report it the way editors number lines.
            throw new SecretsException($"Secrets JSON is malformed (line {(ex.LineNumber ?? 0) + 1}).");
        }
    }
}
