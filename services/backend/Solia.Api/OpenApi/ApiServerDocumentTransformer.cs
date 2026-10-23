using Microsoft.AspNetCore.OpenApi;
using Microsoft.Extensions.Options;
using Microsoft.OpenApi;
using Solia.Core.Configuration;

namespace Solia.Api.OpenApi;

/// <summary>
/// Stamps the generated OpenAPI document with a stable title/version and two servers: first the host that
/// served the document (a relative "/"), which the Swagger UI picks by default so "Try it out" always reaches
/// this instance; then the configured base URL (<see cref="ApiOptions.BaseUrl"/>, from BACKEND_URL) for
/// generated clients — often a compose hostname a browser can't resolve.
/// </summary>
internal sealed class ApiServerDocumentTransformer(IOptions<ApiOptions> options) : IOpenApiDocumentTransformer
{
    public Task TransformAsync(OpenApiDocument document, OpenApiDocumentTransformerContext context, CancellationToken cancellationToken)
    {
        document.Info ??= new OpenApiInfo();
        document.Info.Title = "Solia backend API";
        document.Info.Version = "v1";
        document.Servers =
        [
            new OpenApiServer { Url = "/", Description = "This host" },
            new OpenApiServer { Url = options.Value.BaseUrl, Description = "Configured base URL (BACKEND_URL)" },
        ];
        return Task.CompletedTask;
    }
}
