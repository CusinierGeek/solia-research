using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Core.Tests.Data;

/// <summary>
/// Fast model-shape tests: they build the EF model (no database connection) and assert the mapping over
/// the Sprint-1 schema — table/column names, store types that preserve precision, and the key constraints.
/// </summary>
public class SoliaDbContextModelTests
{
    private static SoliaDbContext BuildContext()
    {
        DbContextOptions<SoliaDbContext> options = new DbContextOptionsBuilder<SoliaDbContext>()
            .UseNpgsql("Host=localhost;Database=solia")
            .Options;
        return new SoliaDbContext(options);
    }

    private static IEntityType Entity<T>(SoliaDbContext ctx) => ctx.Model.FindEntityType(typeof(T))!;

    private static string Column<T>(SoliaDbContext ctx, string propertyName) =>
        Entity<T>(ctx).FindProperty(propertyName)!.GetColumnName();

    [Theory]
    [InlineData(typeof(TokenLive), "tokens_live")]
    [InlineData(typeof(TokenHistory), "tokens_history")]
    [InlineData(typeof(Feature), "features")]
    [InlineData(typeof(Prediction), "predictions")]
    [InlineData(typeof(Trade), "trades")]
    [InlineData(typeof(ModelRecord), "models")]
    [InlineData(typeof(TrainingEvent), "training_events")]
    [InlineData(typeof(LogEntry), "logs")]
    [InlineData(typeof(BotSessionRecord), "bot_sessions")]
    public void Every_entity_maps_to_its_sprint1_table(Type clrType, string table)
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.Equal(table, ctx.Model.FindEntityType(clrType)!.GetTableName());
    }

    [Fact]
    public void All_nine_schema_tables_are_mapped()
    {
        using SoliaDbContext ctx = BuildContext();
        string[] tables = ctx.Model.GetEntityTypes().Select(e => e.GetTableName()!).OrderBy(t => t).ToArray();
        Assert.Equal(
            new[] { "bot_sessions", "features", "logs", "models", "predictions", "tokens_history", "tokens_live", "trades", "training_events" },
            tables);
    }

    [Fact]
    public void Columns_use_the_snake_case_names_from_the_schema()
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.Equal("first_seen_at", Column<TokenHistory>(ctx, nameof(TokenHistory.FirstSeenAt)));
        Assert.Equal("last_market_cap", Column<TokenLive>(ctx, nameof(TokenLive.LastMarketCap)));
        Assert.Equal("buy_sell_ratio", Column<Feature>(ctx, nameof(Feature.BuySellRatio)));
        Assert.Equal("holders_growth_rate", Column<Feature>(ctx, nameof(Feature.HoldersGrowthRate)));
        Assert.Equal("net_sol_flow", Column<Feature>(ctx, nameof(Feature.NetSolFlow)));
        Assert.Equal("large_buy_count", Column<Feature>(ctx, nameof(Feature.LargeBuyCount)));
        Assert.Equal("return_short", Column<Feature>(ctx, nameof(Feature.ReturnShort)));
        Assert.Equal("volume_burst_ratio", Column<Feature>(ctx, nameof(Feature.VolumeBurstRatio)));
        Assert.Equal("price_history_sec", Column<Feature>(ctx, nameof(Feature.PriceHistorySec)));
        Assert.Equal("unique_traders", Column<Feature>(ctx, nameof(Feature.UniqueTraders)));
        Assert.Equal("trader_hhi", Column<Feature>(ctx, nameof(Feature.TraderHhi)));
        Assert.Equal("new_trader_ratio", Column<Feature>(ctx, nameof(Feature.NewTraderRatio)));
        Assert.Equal("burned_liquidity", Column<Feature>(ctx, nameof(Feature.BurnedLiquidity)));
        Assert.Equal("mkt_coverage_sec", Column<Feature>(ctx, nameof(Feature.MktCoverageSec)));
        Assert.Equal("mkt_active_tokens_1h", Column<Feature>(ctx, nameof(Feature.MktActiveTokens1h)));
        Assert.Equal("wal_coverage_sec", Column<Feature>(ctx, nameof(Feature.WalCoverageSec)));
        Assert.Equal("wal_creator_first_sell_sec", Column<Feature>(ctx, nameof(Feature.WalCreatorFirstSellSec)));
        Assert.Equal("pump_prob", Column<Prediction>(ctx, nameof(Prediction.PumpProb)));
        Assert.Equal("features_snapshot", Column<Trade>(ctx, nameof(Trade.FeaturesSnapshot)));
        Assert.Equal("candidate_model_id", Column<TrainingEvent>(ctx, nameof(TrainingEvent.CandidateModelId)));
        Assert.Equal("n_params", Column<ModelRecord>(ctx, nameof(ModelRecord.NParams)));
        Assert.Equal("pruned_at", Column<ModelRecord>(ctx, nameof(ModelRecord.PrunedAt)));
    }

    [Fact]
    public void Natural_and_surrogate_keys_match_the_schema()
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.Equal("mint", Assert.Single(Entity<TokenHistory>(ctx).FindPrimaryKey()!.Properties).GetColumnName());
        Assert.Equal("mint", Assert.Single(Entity<TokenLive>(ctx).FindPrimaryKey()!.Properties).GetColumnName());
        Assert.Equal("id", Assert.Single(Entity<Feature>(ctx).FindPrimaryKey()!.Properties).GetColumnName());
        Assert.Equal("id", Assert.Single(Entity<Trade>(ctx).FindPrimaryKey()!.Properties).GetColumnName());
    }

    [Theory]
    [InlineData(typeof(Feature), nameof(Feature.Price), "numeric(28,24)")]
    [InlineData(typeof(Feature), nameof(Feature.MarketCap), "numeric(28,12)")]
    [InlineData(typeof(TokenLive), nameof(TokenLive.LastPrice), "numeric(28,24)")]
    [InlineData(typeof(Prediction), nameof(Prediction.PositionSize), "numeric(28,12)")]
    [InlineData(typeof(Prediction), nameof(Prediction.StopLoss), "numeric(28,24)")]
    [InlineData(typeof(Trade), nameof(Trade.EntryPrice), "numeric(28,24)")]
    [InlineData(typeof(Trade), nameof(Trade.Pnl), "numeric(28,12)")]
    public void Money_and_price_columns_are_numeric_to_avoid_float_drift(Type clrType, string property, string storeType)
    {
        using SoliaDbContext ctx = BuildContext();
        IProperty p = ctx.Model.FindEntityType(clrType)!.FindProperty(property)!;
        Assert.Equal(storeType, p.GetColumnType());
        Assert.Equal(typeof(decimal), Nullable.GetUnderlyingType(p.ClrType) ?? p.ClrType);
    }

    [Fact]
    public void Every_decimal_column_has_a_precision_that_fits_system_decimal()
    {
        using SoliaDbContext ctx = BuildContext();
        IProperty[] decimals = ctx.Model.GetEntityTypes()
            .SelectMany(e => e.GetProperties())
            .Where(p => (Nullable.GetUnderlyingType(p.ClrType) ?? p.ClrType) == typeof(decimal))
            .ToArray();

        Assert.Equal(17, decimals.Length);
        Assert.All(decimals, p => Assert.Matches(@"^numeric\(28,(12|24)\)$", p.GetColumnType()));
    }

    [Theory]
    [InlineData(typeof(Feature), nameof(Feature.OnchainSignals))]
    [InlineData(typeof(Feature), nameof(Feature.BehavioralSignals))]
    [InlineData(typeof(Feature), nameof(Feature.Raw))]
    [InlineData(typeof(Trade), nameof(Trade.FeaturesSnapshot))]
    [InlineData(typeof(TokenHistory), nameof(TokenHistory.Metadata))]
    [InlineData(typeof(ModelRecord), nameof(ModelRecord.Metrics))]
    [InlineData(typeof(LogEntry), nameof(LogEntry.Context))]
    [InlineData(typeof(BotSessionRecord), nameof(BotSessionRecord.Options))]
    public void Jsonb_columns_map_to_jsonb(Type clrType, string property)
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.Equal("jsonb", ctx.Model.FindEntityType(clrType)!.FindProperty(property)!.GetColumnType());
    }

    [Theory]
    [InlineData(typeof(Feature), nameof(Feature.Ts))]
    [InlineData(typeof(Prediction), nameof(Prediction.Ts))]
    [InlineData(typeof(Trade), nameof(Trade.OpenedAt))]
    [InlineData(typeof(Trade), nameof(Trade.ClosedAt))]
    [InlineData(typeof(TokenHistory), nameof(TokenHistory.FirstSeenAt))]
    [InlineData(typeof(TokenLive), nameof(TokenLive.CreatedAt))]
    public void Timestamptz_columns_map_to_datetimeoffset(Type clrType, string property)
    {
        using SoliaDbContext ctx = BuildContext();
        IProperty p = ctx.Model.FindEntityType(clrType)!.FindProperty(property)!;
        Assert.Equal(typeof(DateTimeOffset), Nullable.GetUnderlyingType(p.ClrType) ?? p.ClrType);
        Assert.Equal("timestamp with time zone", p.GetRelationalTypeMapping().StoreType);
    }

    [Theory]
    [InlineData(typeof(Trade), nameof(Trade.OpenedAt))]
    [InlineData(typeof(Trade), nameof(Trade.ClosedAt))] // nullable: the convention covers DateTimeOffset? too
    public void Datetimeoffsets_are_written_as_utc(Type clrType, string property)
    {
        using SoliaDbContext ctx = BuildContext();
        IProperty p = ctx.Model.FindEntityType(clrType)!.FindProperty(property)!;
        DateTimeOffset local = new(2026, 1, 1, 8, 30, 0, TimeSpan.FromHours(-4));

        DateTimeOffset written = (DateTimeOffset)p.GetTypeMapping().Converter!.ConvertToProvider(local)!;

        Assert.Equal(TimeSpan.Zero, written.Offset);
        Assert.Equal(local, written); // same instant
        Assert.Equal(new DateTime(2026, 1, 1, 12, 30, 0), written.DateTime);
    }

    [Fact]
    public void Doubles_stay_double_precision_for_ratios_and_indicators()
    {
        using SoliaDbContext ctx = BuildContext();
        IProperty pumpProb = Entity<Prediction>(ctx).FindProperty(nameof(Prediction.PumpProb))!;
        Assert.Equal(typeof(double), Nullable.GetUnderlyingType(pumpProb.ClrType) ?? pumpProb.ClrType);
        Assert.Equal("double precision", pumpProb.GetRelationalTypeMapping().StoreType);
        IProperty volatility = Entity<Feature>(ctx).FindProperty(nameof(Feature.Volatility))!;
        Assert.Equal(typeof(double), Nullable.GetUnderlyingType(volatility.ClrType) ?? volatility.ClrType);
        Assert.Equal("double precision", volatility.GetRelationalTypeMapping().StoreType);
    }

    [Theory]
    [InlineData(typeof(TokenLive), nameof(TokenLive.Mint), typeof(TokenHistory), "tokens_live_mint_fkey", DeleteBehavior.Restrict)]
    [InlineData(typeof(Feature), nameof(Feature.Mint), typeof(TokenHistory), "features_mint_fkey", DeleteBehavior.Restrict)]
    [InlineData(typeof(Prediction), nameof(Prediction.Mint), typeof(TokenHistory), "predictions_mint_fkey", DeleteBehavior.Restrict)]
    [InlineData(typeof(Prediction), nameof(Prediction.ModelId), typeof(ModelRecord), "predictions_model_id_fkey", DeleteBehavior.Restrict)]
    [InlineData(typeof(Prediction), nameof(Prediction.FeatureId), typeof(Feature), "predictions_feature_id_fkey", DeleteBehavior.SetNull)]
    [InlineData(typeof(Trade), nameof(Trade.Mint), typeof(TokenHistory), "trades_mint_fkey", DeleteBehavior.Restrict)]
    [InlineData(typeof(Trade), nameof(Trade.PredictionId), typeof(Prediction), "trades_prediction_id_fkey", DeleteBehavior.SetNull)]
    [InlineData(typeof(TrainingEvent), nameof(TrainingEvent.CandidateModelId), typeof(ModelRecord), "training_events_candidate_model_id_fkey", DeleteBehavior.Restrict)]
    [InlineData(typeof(TrainingEvent), nameof(TrainingEvent.ChampionModelId), typeof(ModelRecord), "training_events_champion_model_id_fkey", DeleteBehavior.Restrict)]
    public void Schema_foreign_keys_are_declared(Type dependent, string property, Type principal, string constraint, DeleteBehavior onDelete)
    {
        using SoliaDbContext ctx = BuildContext();
        IForeignKey fk = Assert.Single(
            ctx.Model.FindEntityType(dependent)!.GetForeignKeys(),
            f => f.Properties.Single().Name == property);
        Assert.Equal(principal, fk.PrincipalEntityType.ClrType);
        Assert.Equal(constraint, fk.GetConstraintName());
        Assert.Equal(onDelete, fk.DeleteBehavior);
        Assert.NotNull(fk.DependentToPrincipal); // a reference navigation, so reads can Include it
    }

    [Fact]
    public void Every_sql_foreign_key_is_mapped()
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.Equal(11, ctx.Model.GetEntityTypes().SelectMany(e => e.GetForeignKeys()).Count());
    }

    [Fact]
    public void Bool_columns_start_at_their_sql_defaults()
    {
        // A new instance mirrors a freshly inserted row.
        Assert.True(new TokenLive().IsActive); // DEFAULT TRUE
        Assert.False(new TokenLive().IsMigrated); // DEFAULT FALSE
        Assert.False(new Feature().IsMigrated); // DEFAULT FALSE
        Assert.False(new ModelRecord().IsChampion); // DEFAULT FALSE
    }

    public static TheoryData<object> OneEntityPerTable() =>
    [
        new TokenHistory { Mint = "m" },
        new TokenLive { Mint = "m" },
        new Feature { Mint = "m" },
        new Prediction { Mint = "m", Horizon = "1m" },
        new Trade { Mint = "m", Status = "open" },
        new ModelRecord { Version = "v", Path = "p" },
        new TrainingEvent(),
        new LogEntry { Service = "backend", Level = "info", Message = "x" },
        new BotSessionRecord(),
    ];

    [Theory]
    [MemberData(nameof(OneEntityPerTable))]
    public void Saving_any_entity_is_rejected(object entity)
    {
        using SoliaDbContext ctx = BuildContext();
        ctx.Add(entity);

        InvalidOperationException e = Assert.Throws<InvalidOperationException>(() => ctx.SaveChanges());
        Assert.Contains("read-only", e.Message);
    }

    [Fact]
    public async Task Save_changes_is_rejected_async_and_even_with_nothing_tracked()
    {
        // No-tracking reads leave edits invisible to the change tracker, so SaveChanges must throw
        // unconditionally rather than silently save nothing.
        await using SoliaDbContext ctx = BuildContext();
        Assert.Throws<InvalidOperationException>(() => ctx.SaveChanges());
        await Assert.ThrowsAsync<InvalidOperationException>(() => ctx.SaveChangesAsync());
    }

    [Fact]
    public void A_context_built_outside_di_is_read_only_too()
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.Equal(QueryTrackingBehavior.NoTracking, ctx.ChangeTracker.QueryTrackingBehavior);
    }

    [Fact]
    public void Features_has_the_unique_mint_ts_index()
    {
        using SoliaDbContext ctx = BuildContext();
        IIndex index = Assert.Single(Entity<Feature>(ctx).GetIndexes());
        Assert.True(index.IsUnique);
        Assert.Equal(new[] { "mint", "ts" }, index.Properties.Select(p => p.GetColumnName()));
    }

    [Fact]
    public void Nullable_set_null_fks_are_optional_columns()
    {
        using SoliaDbContext ctx = BuildContext();
        Assert.True(Entity<Prediction>(ctx).FindProperty(nameof(Prediction.FeatureId))!.IsNullable);
        Assert.True(Entity<Trade>(ctx).FindProperty(nameof(Trade.PredictionId))!.IsNullable);
    }

    [Fact]
    public void AddSoliaDbContext_registers_a_read_only_pooled_context()
    {
        IConfiguration config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?> { ["SOLIA_DB_CONNECTION"] = "Host=localhost;Database=solia" })
            .Build();
        ServiceCollection services = new();
        services.AddSingleton(config);
        services.AddSoliaDbContext(config);

        using ServiceProvider provider = services.BuildServiceProvider();
        using IServiceScope scope = provider.CreateScope();

        SoliaDbContext? ctx = scope.ServiceProvider.GetService<SoliaDbContext>();
        Assert.NotNull(ctx);
        Assert.Equal(QueryTrackingBehavior.NoTracking, ctx.ChangeTracker.QueryTrackingBehavior);
        Assert.Throws<InvalidOperationException>(() => ctx.SaveChanges());
    }
}
