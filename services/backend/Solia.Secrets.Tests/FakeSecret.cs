namespace Solia.Secrets.Tests;

/// <summary>
/// Random throwaway values for tests, generated at run time so no secret-looking literal is ever committed
/// (keeps secret scanners such as GitGuardian quiet).
/// </summary>
internal static class FakeSecret
{
    public static string New() => "fake-" + Guid.NewGuid().ToString("N");
}
