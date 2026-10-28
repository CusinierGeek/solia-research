using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using Solia.Core.Data.Entities;

namespace Solia.Core.Data;

/// <summary>
/// EF Core context mapped onto the existing Sprint-1 PostgreSQL schema. The raw SQL migrations in
/// <c>services/db/migrations</c> remain the single source of truth for DDL — this context maps
/// <em>over</em> that schema and must never be used to generate migrations against it.
/// <para>
/// <b>Read-only.</b> Every table is written through raw SQL (the pipeline/bot stores, the log sinks, the
/// Python ai-service), so the schema stays the one contract shared with Python. <see cref="SaveChanges(bool)"/>
/// always throws, and <see cref="ReadOnlyCommandInterceptor"/> rejects any non-<c>SELECT</c> command
/// (<c>ExecuteUpdate</c>/<c>ExecuteDelete</c>/<c>ExecuteSql</c>). Queries default to no-tracking: nothing
/// is ever saved, so tracking would only cost. Paper trading only — no real execution.
/// </para>
/// <para>
/// Foreign keys are declared (with one-way reference navigations) so reads can <c>Include</c> them. Every
/// <see cref="DateTimeOffset"/> query parameter is normalised to UTC (Npgsql rejects a non-zero offset for
/// <c>timestamptz</c>).
/// </para>
/// <para>
/// CHECK constraints (enum-like <c>TEXT</c>, probability/confidence ranges) and the single-champion
/// partial index live in SQL and are intentionally not re-declared here.
/// </para>
/// </summary>
public sealed class SoliaDbContext(DbContextOptions<SoliaDbContext> options) : DbContext(ReadOnly(options))
{
    /// <summary>Price columns (migration 0002): precision 28 always fits a <see cref="decimal"/>.</summary>
    internal const string PriceType = "numeric(28,24)";

    /// <summary>Amount / P&amp;L columns (migration 0002).</summary>
    internal const string AmountType = "numeric(28,12)";

    public DbSet<TokenLive> TokensLive => Set<TokenLive>();

    public DbSet<TokenHistory> TokensHistory => Set<TokenHistory>();

    public DbSet<Feature> Features => Set<Feature>();

    public DbSet<Prediction> Predictions => Set<Prediction>();

    public DbSet<Trade> Trades => Set<Trade>();

    public DbSet<ModelRecord> Models => Set<ModelRecord>();

    public DbSet<TrainingEvent> TrainingEvents => Set<TrainingEvent>();

    public DbSet<LogEntry> Logs => Set<LogEntry>();

    public override int SaveChanges(bool acceptAllChangesOnSuccess) => throw ReadOnly();

    public override Task<int> SaveChangesAsync(bool acceptAllChangesOnSuccess, CancellationToken cancellationToken = default) =>
        throw ReadOnly();


    protected override void ConfigureConventions(ModelConfigurationBuilder configurationBuilder)
    {
        // Also applies to DateTimeOffset? properties.
        configurationBuilder.Properties<DateTimeOffset>().HaveConversion<UtcDateTimeOffsetConverter>();
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<TokenHistory>(e =>
        {
            e.ToTable("tokens_history");
            e.HasKey(x => x.Mint);
            e.Property(x => x.Mint).HasColumnName("mint");
            e.Property(x => x.Symbol).HasColumnName("symbol");
            e.Property(x => x.Name).HasColumnName("name");
            e.Property(x => x.CreatedAt).HasColumnName("created_at");
            e.Property(x => x.FirstSeenAt).HasColumnName("first_seen_at").HasDefaultValueSql("now()");
            e.Property(x => x.LastSeenAt).HasColumnName("last_seen_at");
            e.Property(x => x.FinalOutcome).HasColumnName("final_outcome");
            e.Property(x => x.Metadata).HasColumnName("metadata").HasColumnType("jsonb");
        });

        modelBuilder.Entity<TokenLive>(e =>
        {
            e.ToTable("tokens_live");
            e.HasKey(x => x.Mint);
            e.Property(x => x.Mint).HasColumnName("mint");
            e.Property(x => x.Symbol).HasColumnName("symbol");
            e.Property(x => x.Name).HasColumnName("name");
            e.Property(x => x.CreatedAt).HasColumnName("created_at");
            e.Property(x => x.LastPrice).HasColumnName("last_price").HasColumnType(PriceType);
            e.Property(x => x.LastMarketCap).HasColumnName("last_market_cap").HasColumnType(AmountType);
            e.Property(x => x.LastLiquidity).HasColumnName("last_liquidity").HasColumnType(AmountType);
            e.Property(x => x.Holders).HasColumnName("holders");
            e.Property(x => x.TrendingScore).HasColumnName("trending_score");
            e.Property(x => x.Venue).HasColumnName("venue");
            e.Property(x => x.IsMigrated).HasColumnName("is_migrated");
            e.Property(x => x.IsActive).HasColumnName("is_active");
            e.Property(x => x.UpdatedAt).HasColumnName("updated_at").HasDefaultValueSql("now()");
            e.HasOne(x => x.Token).WithOne().HasForeignKey<TokenLive>(x => x.Mint)
                .HasConstraintName("tokens_live_mint_fkey").OnDelete(DeleteBehavior.Restrict);
        });

        modelBuilder.Entity<Feature>(e =>
        {
            e.ToTable("features");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id");
            e.Property(x => x.Mint).HasColumnName("mint");
            e.Property(x => x.Ts).HasColumnName("ts");
            e.Property(x => x.Venue).HasColumnName("venue");
            e.Property(x => x.IsMigrated).HasColumnName("is_migrated");
            e.Property(x => x.PoolAddress).HasColumnName("pool_address");
            e.Property(x => x.Price).HasColumnName("price").HasColumnType(PriceType);
            e.Property(x => x.Volume).HasColumnName("volume").HasColumnType(AmountType);
            e.Property(x => x.MarketCap).HasColumnName("market_cap").HasColumnType(AmountType);
            e.Property(x => x.Holders).HasColumnName("holders");
            e.Property(x => x.Liquidity).HasColumnName("liquidity").HasColumnType(AmountType);
            e.Property(x => x.BuyEvents).HasColumnName("buy_events");
            e.Property(x => x.SellEvents).HasColumnName("sell_events");
            e.Property(x => x.BuySellRatio).HasColumnName("buy_sell_ratio");
            e.Property(x => x.TokenAgeSec).HasColumnName("token_age_sec");
            e.Property(x => x.TrendingScore).HasColumnName("trending_score");
            e.Property(x => x.SocialMentions).HasColumnName("social_mentions");
            e.Property(x => x.HoldersGrowthRate).HasColumnName("holders_growth_rate");
            e.Property(x => x.VolumeGrowthRate).HasColumnName("volume_growth_rate");
            e.Property(x => x.TxCreationRate).HasColumnName("tx_creation_rate");
            e.Property(x => x.Volatility).HasColumnName("volatility");
            e.Property(x => x.Ema).HasColumnName("ema");
            e.Property(x => x.Rsi).HasColumnName("rsi");
            e.Property(x => x.PriceDerivative).HasColumnName("price_derivative");
            e.Property(x => x.NetSolFlow).HasColumnName("net_sol_flow").HasColumnType(AmountType);
            e.Property(x => x.BuyVolumeRatio).HasColumnName("buy_volume_ratio");
            e.Property(x => x.AvgBuySol).HasColumnName("avg_buy_sol").HasColumnType(AmountType);
            e.Property(x => x.MaxBuySol).HasColumnName("max_buy_sol").HasColumnType(AmountType);
            e.Property(x => x.LargeBuyCount).HasColumnName("large_buy_count");
            e.Property(x => x.ReturnShort).HasColumnName("return_short");
            e.Property(x => x.ReturnMedium).HasColumnName("return_medium");
            e.Property(x => x.ReturnLong).HasColumnName("return_long");
            e.Property(x => x.PriceAcceleration).HasColumnName("price_acceleration");
            e.Property(x => x.EmaSlope).HasColumnName("ema_slope");
            e.Property(x => x.VolumeAcceleration).HasColumnName("volume_acceleration");
            e.Property(x => x.VolumeBurstRatio).HasColumnName("volume_burst_ratio");
            e.Property(x => x.PriceHistorySec).HasColumnName("price_history_sec");
            e.Property(x => x.OnchainSignals).HasColumnName("onchain_signals").HasColumnType("jsonb");
            e.Property(x => x.BehavioralSignals).HasColumnName("behavioral_signals").HasColumnType("jsonb");
            e.Property(x => x.Raw).HasColumnName("raw").HasColumnType("jsonb");
            // Matches CONSTRAINT features_mint_ts_key UNIQUE (mint, ts) in SQL.
            e.HasIndex(x => new { x.Mint, x.Ts }).IsUnique().HasDatabaseName("features_mint_ts_key");
            e.HasOne(x => x.Token).WithMany().HasForeignKey(x => x.Mint)
                .HasConstraintName("features_mint_fkey").OnDelete(DeleteBehavior.Restrict);
        });

        modelBuilder.Entity<Prediction>(e =>
        {
            e.ToTable("predictions");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id");
            e.Property(x => x.Mint).HasColumnName("mint");
            e.Property(x => x.ModelId).HasColumnName("model_id");
            e.Property(x => x.Ts).HasColumnName("ts").HasDefaultValueSql("now()");
            e.Property(x => x.Horizon).HasColumnName("horizon");
            e.Property(x => x.PumpProb).HasColumnName("pump_prob");
            e.Property(x => x.RugProb).HasColumnName("rug_prob");
            e.Property(x => x.Strategy).HasColumnName("strategy");
            e.Property(x => x.PositionSize).HasColumnName("position_size").HasColumnType(AmountType);
            e.Property(x => x.TakeProfit).HasColumnName("take_profit").HasColumnType(PriceType);
            e.Property(x => x.StopLoss).HasColumnName("stop_loss").HasColumnType(PriceType);
            e.Property(x => x.Confidence).HasColumnName("confidence");
            e.Property(x => x.FeatureId).HasColumnName("feature_id");
            e.HasOne(x => x.Token).WithMany().HasForeignKey(x => x.Mint)
                .HasConstraintName("predictions_mint_fkey").OnDelete(DeleteBehavior.Restrict);
            e.HasOne(x => x.Model).WithMany().HasForeignKey(x => x.ModelId)
                .HasConstraintName("predictions_model_id_fkey").OnDelete(DeleteBehavior.Restrict);
            e.HasOne(x => x.Feature).WithMany().HasForeignKey(x => x.FeatureId)
                .HasConstraintName("predictions_feature_id_fkey").OnDelete(DeleteBehavior.SetNull);
        });

        modelBuilder.Entity<Trade>(e =>
        {
            e.ToTable("trades");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id");
            e.Property(x => x.Mint).HasColumnName("mint");
            e.Property(x => x.PredictionId).HasColumnName("prediction_id");
            e.Property(x => x.OpenedAt).HasColumnName("opened_at").HasDefaultValueSql("now()");
            e.Property(x => x.ClosedAt).HasColumnName("closed_at");
            e.Property(x => x.EntryPrice).HasColumnName("entry_price").HasColumnType(PriceType);
            e.Property(x => x.ExitPrice).HasColumnName("exit_price").HasColumnType(PriceType);
            e.Property(x => x.Size).HasColumnName("size").HasColumnType(AmountType);
            e.Property(x => x.DurationSec).HasColumnName("duration_sec");
            e.Property(x => x.Pnl).HasColumnName("pnl").HasColumnType(AmountType);
            e.Property(x => x.PnlPct).HasColumnName("pnl_pct");
            e.Property(x => x.Strategy).HasColumnName("strategy");
            e.Property(x => x.Confidence).HasColumnName("confidence");
            e.Property(x => x.ExitReason).HasColumnName("exit_reason");
            e.Property(x => x.FeaturesSnapshot).HasColumnName("features_snapshot").HasColumnType("jsonb");
            e.Property(x => x.Status).HasColumnName("status").HasDefaultValue("open");
            e.HasOne(x => x.Token).WithMany().HasForeignKey(x => x.Mint)
                .HasConstraintName("trades_mint_fkey").OnDelete(DeleteBehavior.Restrict);
            e.HasOne(x => x.Prediction).WithMany().HasForeignKey(x => x.PredictionId)
                .HasConstraintName("trades_prediction_id_fkey").OnDelete(DeleteBehavior.SetNull);
        });

        modelBuilder.Entity<ModelRecord>(e =>
        {
            e.ToTable("models");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id");
            e.Property(x => x.Version).HasColumnName("version");
            e.Property(x => x.CreatedAt).HasColumnName("created_at").HasDefaultValueSql("now()");
            e.Property(x => x.Path).HasColumnName("path");
            e.Property(x => x.IsChampion).HasColumnName("is_champion");
            e.Property(x => x.Metrics).HasColumnName("metrics").HasColumnType("jsonb");
            e.Property(x => x.Hyperparams).HasColumnName("hyperparams").HasColumnType("jsonb");
            e.Property(x => x.NParams).HasColumnName("n_params");
            e.HasIndex(x => x.Version).IsUnique();
        });

        modelBuilder.Entity<TrainingEvent>(e =>
        {
            e.ToTable("training_events");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id");
            e.Property(x => x.StartedAt).HasColumnName("started_at").HasDefaultValueSql("now()");
            e.Property(x => x.FinishedAt).HasColumnName("finished_at");
            e.Property(x => x.Trigger).HasColumnName("trigger");
            e.Property(x => x.NSamples).HasColumnName("n_samples");
            e.Property(x => x.CandidateModelId).HasColumnName("candidate_model_id");
            e.Property(x => x.ChampionModelId).HasColumnName("champion_model_id");
            e.Property(x => x.Metrics).HasColumnName("metrics").HasColumnType("jsonb");
            e.Property(x => x.Decision).HasColumnName("decision");
            e.Property(x => x.Notes).HasColumnName("notes");
            e.HasOne(x => x.CandidateModel).WithMany().HasForeignKey(x => x.CandidateModelId)
                .HasConstraintName("training_events_candidate_model_id_fkey").OnDelete(DeleteBehavior.Restrict);
            e.HasOne(x => x.ChampionModel).WithMany().HasForeignKey(x => x.ChampionModelId)
                .HasConstraintName("training_events_champion_model_id_fkey").OnDelete(DeleteBehavior.Restrict);
        });

        modelBuilder.Entity<LogEntry>(e =>
        {
            e.ToTable("logs");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id");
            e.Property(x => x.Ts).HasColumnName("ts").HasDefaultValueSql("now()");
            e.Property(x => x.Service).HasColumnName("service");
            e.Property(x => x.Level).HasColumnName("level");
            e.Property(x => x.Message).HasColumnName("message");
            e.Property(x => x.Context).HasColumnName("context").HasColumnType("jsonb");
        });
    }

    // Applied in the constructor (not OnConfiguring, which pooling forbids) so every instance is read-only,
    // however it was built.
    private static DbContextOptions<SoliaDbContext> ReadOnly(DbContextOptions<SoliaDbContext> options) =>
        new DbContextOptionsBuilder<SoliaDbContext>(options)
            .UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking)
            .AddInterceptors(ReadOnlyCommandInterceptor.Instance)
            .Options;

    private static InvalidOperationException ReadOnly() =>
        new("SoliaDbContext is read-only; write through the raw-SQL stores (e.g. the bot's PostgresTradeStore).");

    /// <summary>Sends every <see cref="DateTimeOffset"/> as UTC (same instant); reads are already UTC.</summary>
    private sealed class UtcDateTimeOffsetConverter() : ValueConverter<DateTimeOffset, DateTimeOffset>(
        v => v.ToUniversalTime(),
        v => v);
}
