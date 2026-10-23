namespace Solia.Testing;

/// <summary>
/// xUnit trait names. Integration tests (Testcontainers/Docker) run in CI; locally the fast suite is
/// <c>dotnet test --filter "Category!=Integration"</c>.
/// </summary>
public static class TestCategories
{
    public const string Key = "Category";

    public const string Integration = "Integration";
}
