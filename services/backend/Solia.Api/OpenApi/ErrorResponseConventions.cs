using Solia.Core.Http;

namespace Solia.Api.OpenApi;

/// <summary>
/// Documents the error envelope (<see cref="ApiErrorResponse"/>) on an endpoint or a group, so the OpenAPI
/// document lists every failure status an operation can return with its one shared schema.
/// </summary>
public static class ErrorResponseConventions
{
    /// <summary>
    /// Adds a <see cref="ApiErrorResponse"/> JSON response for each of <paramref name="statusCodes"/> (applied to
    /// a group, it reaches every endpoint in it).
    /// </summary>
    public static TBuilder ProducesError<TBuilder>(this TBuilder builder, params int[] statusCodes)
        where TBuilder : IEndpointConventionBuilder =>
        builder.WithMetadata(statusCodes
            .Select(status => (object)new ProducesResponseTypeMetadata(status, typeof(ApiErrorResponse), ["application/json"]))
            .ToArray());
}
