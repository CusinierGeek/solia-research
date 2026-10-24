using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Solia.Api.Live;

namespace Solia.Api.Tests.Live;

public class LiveHubWiringTests
{
    [Fact]
    public void Hub_payloads_use_a_copy_of_the_rest_serializer_options_so_they_cannot_drift()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
            options.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
        });
        services.AddLiveHub();
        using ServiceProvider provider = services.BuildServiceProvider();

        JsonSerializerOptions hub = provider.GetRequiredService<IOptions<JsonHubProtocolOptions>>().Value.PayloadSerializerOptions;

        Assert.Same(JsonNamingPolicy.SnakeCaseLower, hub.PropertyNamingPolicy);
        Assert.Contains(hub.Converters, c => c is JsonStringEnumConverter);
        Assert.Equal("{\"trade_id\":7,\"side\":\"Short\"}", JsonSerializer.Serialize(new { TradeId = 7, Side = Side.Short }, hub));
    }

    private enum Side
    {
        Long,
        Short,
    }
}
