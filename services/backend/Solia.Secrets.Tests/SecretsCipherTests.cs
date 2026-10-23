using System.Text;

namespace Solia.Secrets.Tests;

public class SecretsCipherTests
{
    private static readonly byte[] Key = MasterKey.Parse(MasterKey.Generate());

    private static readonly Dictionary<string, string> Sample = new()
    {
        [SecretNames.DbPassword] = FakeSecret.New(),
        ["rpc_key"] = FakeSecret.New(),
    };

    [Fact]
    public void Seal_then_open_round_trips_every_secret()
    {
        string sealedContent = SecretsFile.Seal(Sample, Key);

        Assert.Equal(Sample, SecretsFile.Open(sealedContent, Key));
    }

    [Fact]
    public void Envelope_is_nonce_ciphertext_tag_in_base64_and_hides_the_plaintext()
    {
        byte[] plaintext = Encoding.UTF8.GetBytes("hello secrets");

        string sealedContent = SecretsCipher.Encrypt(plaintext, Key);
        byte[] envelope = Convert.FromBase64String(sealedContent);

        Assert.Equal(SecretsCipher.NonceSize + plaintext.Length + SecretsCipher.TagSize, envelope.Length);
        Assert.DoesNotContain("hello", Encoding.UTF8.GetString(envelope), StringComparison.Ordinal);
    }

    [Fact]
    public void Each_encryption_draws_a_fresh_nonce()
    {
        Assert.NotEqual(SecretsFile.Seal(Sample, Key), SecretsFile.Seal(Sample, Key));
    }

    [Theory]
    [InlineData(0)] // nonce
    [InlineData(20)] // ciphertext
    [InlineData(-1)] // tag (last byte)
    public void Any_tampered_byte_fails_authentication(int position)
    {
        byte[] envelope = Convert.FromBase64String(SecretsFile.Seal(Sample, Key));
        envelope[position < 0 ? envelope.Length + position : position] ^= 0x01;

        var ex = Assert.Throws<SecretsException>(() => SecretsFile.Open(Convert.ToBase64String(envelope), Key));
        Assert.Contains("authentication", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Wrong_key_fails_authentication_without_leaking_values()
    {
        string sealedContent = SecretsFile.Seal(Sample, Key);
        byte[] otherKey = MasterKey.Parse(MasterKey.Generate());

        var ex = Assert.Throws<SecretsException>(() => SecretsFile.Open(sealedContent, otherKey));
        Assert.DoesNotContain(Sample[SecretNames.DbPassword], ex.ToString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("not base64 !!")]
    [InlineData("AAAA")] // valid base64, shorter than nonce + tag
    public void Malformed_envelope_is_rejected(string content)
    {
        Assert.Throws<SecretsException>(() => SecretsFile.Open(content, Key));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("   ")]
    [InlineData("not base64 !!")]
    [InlineData("AAAAAAAAAAAAAAAAAAAAAA==")] // 16 bytes, not 32
    public void Master_key_must_be_32_base64_bytes(string? value)
    {
        Assert.Throws<SecretsException>(() => MasterKey.Parse(value));
    }

    [Fact]
    public void Generated_master_key_parses_to_32_bytes()
    {
        Assert.Equal(MasterKey.SizeInBytes, MasterKey.Parse(MasterKey.Generate()).Length);
    }

    [Theory]
    [InlineData("[1, 2]")]
    [InlineData("{\"port\": 5432}")]
    [InlineData("{ broken")]
    public void Secrets_json_must_be_an_object_of_strings(string json)
    {
        var ex = Assert.Throws<SecretsException>(() => SecretsFile.ParseJson(Encoding.UTF8.GetBytes(json)));
        Assert.DoesNotContain("5432", ex.Message, StringComparison.Ordinal);
    }
}
