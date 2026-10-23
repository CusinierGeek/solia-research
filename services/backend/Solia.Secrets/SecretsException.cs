namespace Solia.Secrets;

/// <summary>
/// A secrets file or master key that can't be used: missing, malformed, wrong key or tampered. Messages
/// never carry the key or any decrypted value.
/// </summary>
public sealed class SecretsException(string message, Exception? innerException = null) : Exception(message, innerException);
