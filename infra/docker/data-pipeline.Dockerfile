# SKELETON — implemented in Sprint 6 (docs/sprints/SPRINT-06-docker.md).
# Target: multi-stage .NET 10 build for the services/data-pipeline worker.
#
# FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# WORKDIR /src
# COPY services/data-pipeline/ .
# RUN dotnet publish Solia.DataPipeline -c Release -o /app
#
# FROM mcr.microsoft.com/dotnet/runtime:10.0
# WORKDIR /app
# COPY --from=build /app .
# ENTRYPOINT ["dotnet", "Solia.DataPipeline.dll"]
