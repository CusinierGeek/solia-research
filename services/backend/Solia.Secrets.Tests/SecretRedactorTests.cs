namespace Solia.Secrets.Tests;

public class SecretRedactorTests
{
    [Fact]
    public void Every_known_value_is_masked()
    {
        string first = FakeSecret.New();
        string second = FakeSecret.New();
        var redactor = new SecretRedactor([first, second]);

        string text = redactor.Redact($"login with {first} then call {second} ({first} again)");

        Assert.Equal("login with *** then call *** (*** again)", text);
        Assert.False(redactor.ContainsSecret(text));
    }

    [Fact]
    public void Longer_secret_is_masked_whole_before_a_shorter_one_inside_it()
    {
        var redactor = new SecretRedactor(["abc", "abcdef"]);

        Assert.Equal("x *** y", redactor.Redact("x abcdef y"));
    }

    [Fact]
    public void Redactor_for_a_provider_covers_its_values_and_extras()
    {
        string dbValue = FakeSecret.New();
        string masterKey = MasterKey.Generate();
        var provider = new SecretsProvider(new Dictionary<string, string> { [SecretNames.DbPassword] = dbValue });

        SecretRedactor redactor = SecretRedactor.For(provider, masterKey, null);

        Assert.Equal("*** / ***", redactor.Redact($"{dbValue} / {masterKey}"));
    }

    [Fact]
    public void Empty_values_are_ignored()
    {
        var redactor = new SecretRedactor(["", null]);

        Assert.True(redactor.IsEmpty);
        Assert.Equal("nothing to hide", redactor.Redact("nothing to hide"));
    }

    [Fact]
    public void Redactor_for_a_provider_takes_derived_values_as_extras()
    {
        string password = FakeSecret.New();
        var provider = new SecretsProvider(new Dictionary<string, string>
        {
            [SecretNames.DbConnection] = $"Host=db;Password={password}",
        });

        SecretRedactor redactor = SecretRedactor.For(provider, [password, null]);

        Assert.Equal("auth failed: ***", redactor.Redact($"auth failed: {password}"));
    }

    [Fact]
    public void Text_without_secrets_is_returned_as_is()
    {
        var redactor = new SecretRedactor([FakeSecret.New()]);
        string text = "nothing " + "to hide";

        Assert.Same(text, redactor.Redact(text));
        Assert.False(redactor.ContainsSecret(text));
        Assert.False(redactor.ContainsSecret(null));
    }
}
