using System.Security.Cryptography;

namespace Solia.Secrets;

/// <summary>
/// The one AES-256-GCM implementation, shared by the provider and the <c>solia-secrets</c> CLI
/// (docs/SECRETS.md). Envelope: <c>[12-byte nonce | ciphertext | 16-byte tag]</c>, base64-encoded; a fresh
/// random nonce is drawn for every encryption.
/// </summary>
public static class SecretsCipher
{
    public const int NonceSize = 12;
    public const int TagSize = 16;

    public static string Encrypt(ReadOnlySpan<byte> plaintext, ReadOnlySpan<byte> key)
    {
        byte[] envelope = new byte[NonceSize + plaintext.Length + TagSize];
        Span<byte> nonce = envelope.AsSpan(0, NonceSize);
        Span<byte> ciphertext = envelope.AsSpan(NonceSize, plaintext.Length);
        Span<byte> tag = envelope.AsSpan(NonceSize + plaintext.Length, TagSize);

        RandomNumberGenerator.Fill(nonce);
        using var aes = new AesGcm(CheckKey(key), TagSize);
        aes.Encrypt(nonce, plaintext, ciphertext, tag);
        return Convert.ToBase64String(envelope);
    }

    /// <summary>
    /// Authenticates, then decrypts. A wrong key or any altered byte throws <see cref="SecretsException"/>
    /// and no plaintext — partial or otherwise — is returned.
    /// </summary>
    public static byte[] Decrypt(string base64Envelope, ReadOnlySpan<byte> key)
    {
        byte[] envelope;
        try
        {
            envelope = Convert.FromBase64String(base64Envelope.Trim());
        }
        catch (FormatException ex)
        {
            throw new SecretsException("Secrets file is not valid base64.", ex);
        }

        if (envelope.Length < NonceSize + TagSize)
        {
            throw new SecretsException("Secrets file is too short to be an AES-GCM envelope.");
        }

        int ciphertextLength = envelope.Length - NonceSize - TagSize;
        byte[] plaintext = new byte[ciphertextLength];
        using var aes = new AesGcm(CheckKey(key), TagSize);
        try
        {
            aes.Decrypt(
                envelope.AsSpan(0, NonceSize),
                envelope.AsSpan(NonceSize, ciphertextLength),
                envelope.AsSpan(NonceSize + ciphertextLength, TagSize),
                plaintext);
        }
        catch (CryptographicException ex)
        {
            // Belt and braces: AesGcm already clears the output on a tag mismatch.
            CryptographicOperations.ZeroMemory(plaintext);
            throw new SecretsException("Secrets file failed authentication (wrong master key or tampered file).", ex);
        }
        return plaintext;
    }

    private static ReadOnlySpan<byte> CheckKey(ReadOnlySpan<byte> key) =>
        key.Length == MasterKey.SizeInBytes
            ? key
            : throw new SecretsException($"Master key must be {MasterKey.SizeInBytes} bytes.");
}
