# SKELETON — implemented in Sprint 6 (docs/sprints/SPRINT-06-docker.md).
# Target: multi-stage .NET 10 build for services/dashboard (Blazor Server).
#
# FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# WORKDIR /src
# COPY services/dashboard/ .
# RUN dotnet publish Solia.Dashboard -c Release -o /app
#
# FROM mcr.microsoft.com/dotnet/aspnet:10.0
# WORKDIR /app
# COPY --from=build /app .
# EXPOSE 8081
# ENTRYPOINT ["dotnet", "Solia.Dashboard.dll"]
