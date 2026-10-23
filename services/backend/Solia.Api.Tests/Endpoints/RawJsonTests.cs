using System.Text.Json;
using Solia.Api.Endpoints;

namespace Solia.Api.Tests.Endpoints;

public class RawJsonTests
{
    private static readonly JsonSerializerOptions Options = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    [Fact]
    public void Is_written_verbatim_as_json_not_as_a_string()
    {
        var log = new LogDto(1, DateTimeOffset.UnixEpoch, "bot", "info", "m", new RawJson("{\"events\": 42}"));

        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(log, Options));

        JsonElement context = doc.RootElement.GetProperty("context");
        Assert.Equal(JsonValueKind.Object, context.ValueKind);
        Assert.Equal(42, context.GetProperty("events").GetInt32());
    }

    [Fact]
    public void Absent_context_is_null()
    {
        var log = new LogDto(1, DateTimeOffset.UnixEpoch, "bot", "info", "m", null);

        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(log, Options));

        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("context").ValueKind);
    }

    [Fact]
    public void Reads_back_the_raw_value()
    {
        RawJson value = JsonSerializer.Deserialize<RawJson>("[1, {\"a\": true}]");

        Assert.Equal("[1, {\"a\": true}]", value.Json);
    }
}
