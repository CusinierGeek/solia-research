namespace Solia.Secrets.Tests;

public class SecretsProviderTests
{
    private static Func<string, string?> Lookup(params (string Key, string? Value)[] values) =>
        key => values.FirstOrDefault(v => v.Key == key).Value;

    [Fact]
    public void Load_decrypts_the_configured_file_into_memory()
    {
        using var dir = new TempDir();
        string key = MasterKey.Generate();
        string path = dir.File("secrets.enc");
        string dbValue = FakeSecret.New();
        SecretsFile.Write(path, new Dictionary<string, string> { [SecretNames.DbPassword] = dbValue }, MasterKey.Parse(key));

        SecretsProvider provider = SecretsProvider.Load(Lookup(
            (SecretsProvider.FileKey, path), (MasterKey.EnvironmentKey, key)));

        Assert.Equal(dbValue, provider.Get(SecretNames.DbPassword));
        Assert.Equal([SecretNames.DbPassword], provider.Names);
    }

    [Fact]
    public void Load_without_a_file_at_the_default_path_is_empty_so_plaintext_config_applies()
    {
        Assert.False(File.Exists(SecretsFile.DefaultPath));

        SecretsProvider provider = SecretsProvider.Load(Lookup());

        Assert.Empty(provider.Names);
    }

    [Theory]
    [InlineData("absent.enc")]
    [InlineData(" absent.enc ")]
    public void Load_fails_fast_when_an_explicit_file_is_missing(string name)
    {
        using var dir = new TempDir();
        string path = Path.Combine(dir.Path, name);

        var ex = Assert.Throws<SecretsException>(() => SecretsProvider.Load(Lookup((SecretsProvider.FileKey, path))));

        Assert.Contains(SecretsProvider.FileKey, ex.Message, StringComparison.Ordinal);
        Assert.Contains("absent.enc", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_fails_fast_when_the_file_exists_but_the_key_is_missing()
    {
        using var dir = new TempDir();
        string path = dir.File("secrets.enc");
        SecretsFile.Write(path, new Dictionary<string, string> { ["a"] = "b" }, MasterKey.Parse(MasterKey.Generate()));

        var ex = Assert.Throws<SecretsException>(() => SecretsProvider.Load(Lookup((SecretsProvider.FileKey, path))));
        Assert.Contains(MasterKey.EnvironmentKey, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_fails_fast_on_a_wrong_key()
    {
        using var dir = new TempDir();
        string path = dir.File("secrets.enc");
        SecretsFile.Write(path, new Dictionary<string, string> { ["a"] = "b" }, MasterKey.Parse(MasterKey.Generate()));

        Assert.Throws<SecretsException>(() => SecretsProvider.Load(Lookup(
            (SecretsProvider.FileKey, path), (MasterKey.EnvironmentKey, MasterKey.Generate()))));
    }

    [Fact]
    public void Get_of_an_undefined_secret_names_it()
    {
        var ex = Assert.Throws<SecretsException>(() => SecretsProvider.Empty.Get("rpc_key"));

        Assert.Contains("rpc_key", ex.Message, StringComparison.Ordinal);
        Assert.False(SecretsProvider.Empty.TryGet("rpc_key", out _));
    }
}
