using Solia.Secrets.Cli;

namespace Solia.Secrets.Tests;

public sealed class SecretsCliTests : IDisposable
{
    private readonly TempDir _dir = new();
    private readonly string _key = MasterKey.Generate();
    private readonly StringWriter _out = new();
    private readonly StringWriter _err = new();
    private readonly string _dbValue = FakeSecret.New();
    private readonly string _rpcValue = FakeSecret.New();

    public void Dispose() => _dir.Dispose();

    private int Run(Func<string, string?> env, params string[] args) => SecretsCli.Run(args, _out, _err, env);

    private int Run(params string[] args) => Run(Env(_key), args);

    private static Func<string, string?> Env(string? key) =>
        name => name == MasterKey.EnvironmentKey ? key : null;

    private string EncryptSample()
    {
        string plain = _dir.File("secrets.plain.json");
        string enc = _dir.File("secrets.enc");
        File.WriteAllText(plain, $$"""{ "db_password": "{{_dbValue}}", "rpc_key": "{{_rpcValue}}" }""");
        Assert.Equal(SecretsCli.Ok, Run("encrypt", "--in", plain, "--out", enc));
        return enc;
    }

    [Fact]
    public void Encrypt_writes_a_sealed_file_the_provider_can_read()
    {
        string enc = EncryptSample();

        Assert.DoesNotContain(_dbValue, File.ReadAllText(enc), StringComparison.Ordinal);
        Assert.Equal(_dbValue, SecretsFile.Read(enc, MasterKey.Parse(_key))["db_password"]);
        Assert.DoesNotContain(_dbValue, _out.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Get_prints_the_named_secret()
    {
        string enc = EncryptSample();

        Assert.Equal(SecretsCli.Ok, Run("get", "--name", "rpc_key", "--file", enc));
        Assert.EndsWith(_rpcValue + Environment.NewLine, _out.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Get_of_an_unknown_name_fails_and_names_it()
    {
        string enc = EncryptSample();

        Assert.Equal(SecretsCli.Failure, Run("get", "--name", "nope", "--file", enc));
        Assert.Contains("'nope'", _err.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Get_without_the_master_key_fails()
    {
        string enc = EncryptSample();

        Assert.Equal(SecretsCli.Failure, Run(Env(null), "get", "--name", "rpc_key", "--file", enc));
        Assert.Contains(MasterKey.EnvironmentKey, _err.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Get_with_a_wrong_master_key_fails_authentication()
    {
        string enc = EncryptSample();

        Assert.Equal(SecretsCli.Failure, Run(Env(MasterKey.Generate()), "get", "--name", "rpc_key", "--file", enc));
        Assert.DoesNotContain(_rpcValue, _out.ToString() + _err.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Rotate_reencrypts_with_the_new_key_and_preserves_values()
    {
        string enc = EncryptSample();
        string newKey = MasterKey.Generate();

        Assert.Equal(SecretsCli.Ok, Run("rotate", "--old-key", _key, "--new-key", newKey, "--file", enc));

        Assert.Throws<SecretsException>(() => SecretsFile.Read(enc, MasterKey.Parse(_key)));
        Dictionary<string, string> rotated = SecretsFile.Read(enc, MasterKey.Parse(newKey));
        Assert.Equal(_dbValue, rotated["db_password"]);
        Assert.Equal(_rpcValue, rotated["rpc_key"]);
    }

    [Fact]
    public void Rotate_with_a_wrong_old_key_leaves_the_file_untouched()
    {
        string enc = EncryptSample();
        string before = File.ReadAllText(enc);

        Assert.Equal(SecretsCli.Failure,
            Run("rotate", "--old-key", MasterKey.Generate(), "--new-key", MasterKey.Generate(), "--file", enc));
        Assert.Equal(before, File.ReadAllText(enc));
    }

    [Fact]
    public void Rotate_with_an_invalid_new_key_leaves_the_file_untouched()
    {
        string enc = EncryptSample();
        string before = File.ReadAllText(enc);

        Assert.Equal(SecretsCli.Failure, Run("rotate", "--old-key", _key, "--new-key", "short", "--file", enc));
        Assert.Equal(before, File.ReadAllText(enc));
    }

    [Fact]
    public void Encrypt_of_a_missing_input_fails()
    {
        Assert.Equal(SecretsCli.Failure,
            Run("encrypt", "--in", _dir.File("absent.json"), "--out", _dir.File("out.enc")));
        Assert.False(File.Exists(_dir.File("out.enc")));
    }

    [Theory]
    [InlineData]
    [InlineData("frobnicate")]
    [InlineData("get")]
    [InlineData("get", "--name")]
    [InlineData("get", "name", "x")]
    [InlineData("get", "--name", "x", "--fiel", "other.enc")]
    [InlineData("get", "--name", "x", "--name", "y")]
    [InlineData("keygen", "--out", "x")]
    public void Bad_usage_returns_the_usage_code(params string[] args)
    {
        Assert.Equal(SecretsCli.Usage, Run(args));
        Assert.Contains("Usage:", _err.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Rotate_with_a_repeated_file_option_touches_neither_file()
    {
        string enc = EncryptSample();
        string before = File.ReadAllText(enc);

        Assert.Equal(SecretsCli.Usage, Run(
            "rotate", "--old-key", _key, "--new-key", MasterKey.Generate(), "--file", enc, "--file", _dir.File("b.enc")));

        Assert.Equal(before, File.ReadAllText(enc));
        Assert.False(File.Exists(_dir.File("b.enc")));
        Assert.Contains("--file", _err.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void A_mistyped_option_is_named_in_the_error()
    {
        Assert.Equal(SecretsCli.Usage, Run("get", "--name", "rpc_key", "--fiel", "x.enc"));
        Assert.Contains("'--fiel'", _err.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Encrypt_accepts_a_plain_file_saved_with_a_bom()
    {
        string plain = _dir.File("bom.plain.json");
        string enc = _dir.File("bom.enc");
        File.WriteAllText(plain, $$"""{ "rpc_key": "{{_rpcValue}}" }""", new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

        Assert.Equal(SecretsCli.Ok, Run("encrypt", "--in", plain, "--out", enc));
        Assert.Equal(_rpcValue, SecretsFile.Read(enc, MasterKey.Parse(_key))["rpc_key"]);
    }

    [Fact]
    public void Keygen_prints_a_valid_master_key()
    {
        Assert.Equal(SecretsCli.Ok, Run("keygen"));
        Assert.Equal(MasterKey.SizeInBytes, MasterKey.Parse(_out.ToString()).Length);
    }
}
