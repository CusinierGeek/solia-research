# SKELETON — implemented in Sprint 6 (docs/sprints/SPRINT-06-docker.md).
# Target: multi-stage .NET 10 build for services/backend.
#
# FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# WORKDIR /src
# COPY services/backend/ .
# RUN dotnet publish Solia.Api -c Release -o /app
#
# FROM mcr.microsoft.com/dotnet/aspnet:10.0
# WORKDIR /app
# COPY --from=build /app .
# EXPOSE 8080
# ENTRYPOINT ["dotnet", "Solia.Api.dll"]
