using System.Text;

namespace Solia.Secrets.Tests;

public class SecretsFileTests
{
    private static readonly byte[] Key = MasterKey.Parse(MasterKey.Generate());

    [Fact]
    public void A_leading_utf8_bom_is_accepted()
    {
        string value = FakeSecret.New();
        byte[] json = [0xEF, 0xBB, 0xBF, .. Encoding.UTF8.GetBytes($$"""{ "rpc_key": "{{value}}" }""")];

        Assert.Equal(value, SecretsFile.ParseJson(json)["rpc_key"]);
    }

    [Theory]
    [InlineData("{ \"a\": \"b\" } garbage")]
    [InlineData("{ \"a\": \"b\" }{ \"c\": \"d\" }")]
    public void Content_after_the_object_is_rejected(string json)
    {
        Assert.Throws<SecretsException>(() => SecretsFile.ParseJson(Encoding.UTF8.GetBytes(json)));
    }

    [Fact]
    public void Trailing_whitespace_is_fine()
    {
        Assert.Equal("b", SecretsFile.ParseJson(Encoding.UTF8.GetBytes("{ \"a\": \"b\" }\r\n\r\n"))["a"]);
    }

    [Fact]
    public void Malformed_json_reports_a_one_based_line()
    {
        byte[] json = Encoding.UTF8.GetBytes("{\n  \"a\": \"b\",\n  oops\n}");

        var ex = Assert.Throws<SecretsException>(() => SecretsFile.ParseJson(json));

        Assert.Contains("line 3", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Write_leaves_no_temp_file_behind()
    {
        using var dir = new TempDir();
        string path = dir.File("secrets.enc");

        SecretsFile.Write(path, new Dictionary<string, string> { ["a"] = "b" }, Key);
        SecretsFile.Write(path, new Dictionary<string, string> { ["a"] = "c" }, Key);

        Assert.Equal([path], Directory.GetFiles(dir.Path));
        Assert.Equal("c", SecretsFile.Read(path, Key)["a"]);
    }

    [Fact]
    public void A_failed_write_removes_its_temp_file()
    {
        using var dir = new TempDir();
        // A directory where the file should go: the final rename fails after the temp file is written.
        string path = Directory.CreateDirectory(dir.File("secrets.enc")).FullName;

        Assert.ThrowsAny<Exception>(() => SecretsFile.Write(path, new Dictionary<string, string> { ["a"] = "b" }, Key));

        Assert.Empty(Directory.GetFiles(dir.Path));
    }
}
