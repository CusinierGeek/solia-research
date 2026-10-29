using System.Text.Json;
using Npgsql;
using NpgsqlTypes;
using Solia.DataPipeline.Features;
using static Solia.Shared.Data.NpgsqlParameters;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// Persists a flush to PostgreSQL in one transaction of at most four set-based statements (one
/// array parameter per column, expanded with <c>unnest</c>), however many tokens it carries:
/// <list type="number">
/// <item>mark deactivated mints inactive in <c>tokens_live</c>;</item>
/// <item>upsert <c>tokens_history</c> (FK target), only ever widening <c>first_seen_at</c> /
/// <c>last_seen_at</c> and filling a missing <c>created_at</c>;</item>
/// <item>upsert the hot <c>tokens_live</c> row only when the snapshot is at least as new as the
/// stored one, so replaying an older window never rolls live data back in time;</item>
/// <item>insert <c>features</c> <c>ON CONFLICT (mint, ts) DO NOTHING</c> (idempotent re-runs), or — in
/// rebuild mode — <c>DO UPDATE</c> every column, so a re-replay refreshes rows written before new columns existed.
/// Rebuild only refreshes snapshots of tokens whose creation this run observed (<see cref="FeatureSnapshot.CreatedAtUtc"/>
/// set): their whole history is in the run. A token born before the replay window starts is tracked from an empty
/// state, so its holders, unique traders, growth rates and momentum would be computed from a truncated history and
/// would degrade correct stored rows; its existing rows are kept (missing ones are still inserted).</item>
/// </list>
/// Snapshots in one flush have distinct mints (one per tracked token).
/// </summary>
public sealed class PostgresTokenSink : ITokenSink
{
    private const string DeactivateSql =
        "UPDATE tokens_live SET is_active = FALSE, updated_at = @ts " +
        "WHERE mint = ANY(@mints) AND is_active AND updated_at <= @ts";

    private const string TokensHistorySql =
        "INSERT INTO tokens_history (mint, created_at, first_seen_at, last_seen_at) " +
        "SELECT * FROM unnest(@mint, @created, @first, @ts) " +
        "ON CONFLICT (mint) DO UPDATE SET " +
        "created_at = COALESCE(tokens_history.created_at, EXCLUDED.created_at), " +
        "first_seen_at = LEAST(tokens_history.first_seen_at, EXCLUDED.first_seen_at), " +
        "last_seen_at = GREATEST(tokens_history.last_seen_at, EXCLUDED.last_seen_at)";

    private const string TokensLiveSql =
        "INSERT INTO tokens_live " +
        "(mint, created_at, last_price, last_market_cap, last_liquidity, holders, trending_score, venue, is_migrated, is_active, updated_at) " +
        "SELECT m, c, p, mc, liq, h, tr, v, mig, TRUE, t " +
        "FROM unnest(@mint, @created, @price, @mc, @liq, @holders, @trending, @venue, @migrated, @ts) " +
        "AS u(m, c, p, mc, liq, h, tr, v, mig, t) " +
        "ON CONFLICT (mint) DO UPDATE SET " +
        "created_at = COALESCE(tokens_live.created_at, EXCLUDED.created_at), " +
        "last_price = EXCLUDED.last_price, last_market_cap = EXCLUDED.last_market_cap, " +
        "last_liquidity = EXCLUDED.last_liquidity, holders = EXCLUDED.holders, trending_score = EXCLUDED.trending_score, " +
        "venue = EXCLUDED.venue, is_migrated = tokens_live.is_migrated OR EXCLUDED.is_migrated, " +
        "is_active = TRUE, updated_at = EXCLUDED.updated_at " +
        "WHERE tokens_live.updated_at <= EXCLUDED.updated_at";

    private const string FeatureColumns =
        "mint, ts, venue, is_migrated, pool_address, price, volume, market_cap, holders, liquidity, " +
        "buy_events, sell_events, buy_sell_ratio, token_age_sec, trending_score, social_mentions, " +
        "holders_growth_rate, volume_growth_rate, tx_creation_rate, volatility, ema, rsi, price_derivative, " +
        "net_sol_flow, buy_volume_ratio, avg_buy_sol, max_buy_sol, large_buy_count, " +
        "return_short, return_medium, return_long, price_acceleration, ema_slope, volume_acceleration, volume_burst_ratio, " +
        "price_history_sec, unique_traders, top_trader_share, trader_hhi, new_trader_ratio, burned_liquidity, " +
        "onchain_signals, behavioral_signals";

    private const string FeaturesInsertSql =
        "INSERT INTO features (" + FeatureColumns + ") " +
        "SELECT * FROM unnest(@mint, @ts, @venue, @migrated, @pool, @price, @volume, @mc, @holders, @liq, " +
        "@buys, @sells, @ratio, @age, @trending, @social, @hgrowth, @vgrowth, @txrate, @vol, @ema, @rsi, @deriv, " +
        "@netflow, @buyvolratio, @avgbuy, @maxbuy, @largebuys, " +
        "@ret_s, @ret_m, @ret_l, @paccel, @emaslope, @vaccel, @burst, @pricehist, " +
        "@uniqtraders, @toptrader, @hhi, @newtraders, @burned, @onchain, @behavioral) ";

    internal const string FeaturesInsertOnlySql = FeaturesInsertSql + "ON CONFLICT (mint, ts) DO NOTHING";

    /// <summary>Rebuild mode: every non-key column of an existing (mint, ts) row is overwritten.</summary>
    internal static readonly string FeaturesRefreshSql =
        FeaturesInsertSql + "ON CONFLICT (mint, ts) DO UPDATE SET " + string.Join(", ",
            FeatureColumns.Split(',', StringSplitOptions.TrimEntries)
                .Where(c => c is not ("mint" or "ts"))
                .Select(c => $"{c} = EXCLUDED.{c}"));

    private readonly NpgsqlDataSource _dataSource;
    private readonly bool _refreshExistingFeatures;
    private long _featureRowsWritten;

    public PostgresTokenSink(NpgsqlDataSource dataSource)
        : this(dataSource, refreshExistingFeatures: false)
    {
    }

    /// <param name="refreshExistingFeatures">
    /// False (live, default replay): a <c>features</c> row already stored for a (mint, ts) is kept as-is.
    /// True (replay <c>--rebuild</c>): it is overwritten, so columns added since it was written get filled —
    /// only for tokens created inside the run (see the class remarks).
    /// </param>
    public PostgresTokenSink(NpgsqlDataSource dataSource, bool refreshExistingFeatures)
    {
        _dataSource = dataSource;
        _refreshExistingFeatures = refreshExistingFeatures;
    }

    /// <summary>
    /// Rebuild mode only overwrites a stored row when the token's full history is in the run, i.e. its
    /// creation was observed; otherwise the recomputed snapshot is incomplete.
    /// </summary>
    internal static bool CanRefresh(FeatureSnapshot snapshot) => snapshot.CreatedAtUtc is not null;

    /// <summary><c>features</c> rows inserted (or refreshed, in rebuild mode) by committed writes.</summary>
    public long FeatureRowsWritten => Interlocked.Read(ref _featureRowsWritten);

    public async Task WriteAsync(FlushBatch batch, CancellationToken cancellationToken)
    {
        if (batch.IsEmpty)
        {
            return;
        }

        await using NpgsqlConnection connection = await _dataSource.OpenConnectionAsync(cancellationToken);
        await using NpgsqlTransaction transaction = await connection.BeginTransactionAsync(cancellationToken);
        await using var commands = new NpgsqlBatch(connection, transaction);
        var features = new List<NpgsqlBatchCommand>(2);

        // Deactivate first, so a mint dropped and re-tracked within one interval ends up active.
        if (batch.Deactivated.Count > 0)
        {
            commands.BatchCommands.Add(Command(DeactivateSql,
                Param("ts", NpgsqlDbType.TimestampTz, batch.Ts),
                Array("mints", NpgsqlDbType.Text, batch.Deactivated.ToArray())));
        }

        IReadOnlyList<FeatureSnapshot> s = batch.Snapshots;
        if (s.Count > 0)
        {
            string[] mints = Column(s, x => x.Mint);
            DateTimeOffset[] ts = Column(s, x => x.Ts);
            DateTimeOffset?[] created = Column(s, x => x.CreatedAtUtc);

            commands.BatchCommands.Add(Command(TokensHistorySql,
                Array("mint", NpgsqlDbType.Text, mints),
                Array("created", NpgsqlDbType.TimestampTz, created),
                Array("first", NpgsqlDbType.TimestampTz, Column(s, x => x.FirstSeenUtc)),
                Array("ts", NpgsqlDbType.TimestampTz, ts)));

            commands.BatchCommands.Add(Command(TokensLiveSql,
                Array("mint", NpgsqlDbType.Text, mints),
                Array("created", NpgsqlDbType.TimestampTz, created),
                Array("price", NpgsqlDbType.Numeric, Column(s, x => x.Price)),
                Array("mc", NpgsqlDbType.Numeric, Column(s, x => x.MarketCap)),
                Array("liq", NpgsqlDbType.Numeric, Column(s, x => x.Liquidity)),
                Array("holders", NpgsqlDbType.Integer, Column(s, x => x.Holders)),
                Array("trending", NpgsqlDbType.Double, Column(s, x => x.TrendingScore)),
                Array("venue", NpgsqlDbType.Text, Column(s, x => x.Venue)),
                Array("migrated", NpgsqlDbType.Boolean, Column(s, x => x.IsMigrated)),
                Array("ts", NpgsqlDbType.TimestampTz, ts)));

            if (_refreshExistingFeatures)
            {
                AddFeatures(s.Where(CanRefresh).ToList(), FeaturesRefreshSql);
                AddFeatures(s.Where(x => !CanRefresh(x)).ToList(), FeaturesInsertOnlySql);
            }
            else
            {
                AddFeatures(s, FeaturesInsertOnlySql);
            }
        }

        await commands.ExecuteNonQueryAsync(cancellationToken);
        await transaction.CommitAsync(cancellationToken);
        Interlocked.Add(ref _featureRowsWritten, features.Sum(f => (long)f.RecordsAffected));

        void AddFeatures(IReadOnlyList<FeatureSnapshot> rows, string sql)
        {
            if (rows.Count > 0)
            {
                NpgsqlBatchCommand command = FeaturesCommand(rows, sql);
                features.Add(command);
                commands.BatchCommands.Add(command);
            }
        }
    }

    /// <summary>
    /// Leaves <c>updated_at</c> untouched: the row's data hasn't changed, and the newer-wins guard
    /// on the live upsert must keep comparing snapshot times.
    /// </summary>
    public async Task DeactivateAllAsync(CancellationToken cancellationToken)
    {
        await using NpgsqlCommand cmd = _dataSource.CreateCommand("UPDATE tokens_live SET is_active = FALSE WHERE is_active");
        await cmd.ExecuteNonQueryAsync(cancellationToken);
    }

    private static NpgsqlBatchCommand FeaturesCommand(IReadOnlyList<FeatureSnapshot> s, string sql)
    {
        string[] mints = Column(s, x => x.Mint);
        DateTimeOffset[] ts = Column(s, x => x.Ts);
        return Command(sql,
            Array("mint", NpgsqlDbType.Text, mints),
            Array("ts", NpgsqlDbType.TimestampTz, ts),
            Array("venue", NpgsqlDbType.Text, Column(s, x => x.Venue)),
            Array("migrated", NpgsqlDbType.Boolean, Column(s, x => x.IsMigrated)),
            Array("pool", NpgsqlDbType.Text, Column(s, x => x.PoolAddress)),
            Array("price", NpgsqlDbType.Numeric, Column(s, x => x.Price)),
            Array("volume", NpgsqlDbType.Numeric, Column(s, x => x.Volume)),
            Array("mc", NpgsqlDbType.Numeric, Column(s, x => x.MarketCap)),
            Array("holders", NpgsqlDbType.Integer, Column(s, x => x.Holders)),
            Array("liq", NpgsqlDbType.Numeric, Column(s, x => x.Liquidity)),
            Array("buys", NpgsqlDbType.Integer, Column(s, x => x.BuyEvents)),
            Array("sells", NpgsqlDbType.Integer, Column(s, x => x.SellEvents)),
            Array("ratio", NpgsqlDbType.Double, Column(s, x => x.BuySellRatio)),
            Array("age", NpgsqlDbType.Integer, Column(s, x => x.TokenAgeSec)),
            Array("trending", NpgsqlDbType.Double, Column(s, x => x.TrendingScore)),
            Array("social", NpgsqlDbType.Integer, Column(s, x => x.SocialMentions)),
            Array("hgrowth", NpgsqlDbType.Double, Column(s, x => x.HoldersGrowthRate)),
            Array("vgrowth", NpgsqlDbType.Double, Column(s, x => x.VolumeGrowthRate)),
            Array("txrate", NpgsqlDbType.Double, Column(s, x => x.TxCreationRate)),
            Array("vol", NpgsqlDbType.Double, Column(s, x => x.Volatility)),
            Array("ema", NpgsqlDbType.Double, Column(s, x => x.Ema)),
            Array("rsi", NpgsqlDbType.Double, Column(s, x => x.Rsi)),
            Array("deriv", NpgsqlDbType.Double, Column(s, x => x.PriceDerivative)),
            Array("netflow", NpgsqlDbType.Numeric, Column(s, x => x.NetSolFlow)),
            Array("buyvolratio", NpgsqlDbType.Double, Column(s, x => x.BuyVolumeRatio)),
            Array("avgbuy", NpgsqlDbType.Numeric, Column(s, x => x.AvgBuySol)),
            Array("maxbuy", NpgsqlDbType.Numeric, Column(s, x => x.MaxBuySol)),
            Array("largebuys", NpgsqlDbType.Integer, Column(s, x => x.LargeBuyCount)),
            Array("ret_s", NpgsqlDbType.Double, Column(s, x => x.ReturnShort)),
            Array("ret_m", NpgsqlDbType.Double, Column(s, x => x.ReturnMedium)),
            Array("ret_l", NpgsqlDbType.Double, Column(s, x => x.ReturnLong)),
            Array("paccel", NpgsqlDbType.Double, Column(s, x => x.PriceAcceleration)),
            Array("emaslope", NpgsqlDbType.Double, Column(s, x => x.EmaSlope)),
            Array("vaccel", NpgsqlDbType.Double, Column(s, x => x.VolumeAcceleration)),
            Array("burst", NpgsqlDbType.Double, Column(s, x => x.VolumeBurstRatio)),
            Array("pricehist", NpgsqlDbType.Double, Column(s, x => x.PriceHistorySec)),
            Array("uniqtraders", NpgsqlDbType.Integer, Column(s, x => x.UniqueTraders)),
            Array("toptrader", NpgsqlDbType.Double, Column(s, x => x.TopTraderShare)),
            Array("hhi", NpgsqlDbType.Double, Column(s, x => x.TraderHhi)),
            Array("newtraders", NpgsqlDbType.Double, Column(s, x => x.NewTraderRatio)),
            Array("burned", NpgsqlDbType.Double, Column(s, x => x.BurnedLiquidity)),
            Array("onchain", NpgsqlDbType.Jsonb, Column(s, x => JsonSerializer.Serialize(x.OnchainSignals))),
            Array("behavioral", NpgsqlDbType.Jsonb, Column(s, x => JsonSerializer.Serialize(x.BehavioralSignals))));
    }

    private static T[] Column<T>(IReadOnlyList<FeatureSnapshot> snapshots, Func<FeatureSnapshot, T> select)
    {
        var values = new T[snapshots.Count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = select(snapshots[i]);
        }

        return values;
    }

    private static NpgsqlBatchCommand Command(string sql, params NpgsqlParameter[] parameters)
    {
        var cmd = new NpgsqlBatchCommand(sql);
        cmd.Parameters.AddRange(parameters);
        return cmd;
    }

    // Explicitly typed so arrays containing NULLs don't trip type inference.
    private static NpgsqlParameter Array(string name, NpgsqlDbType elementType, System.Array values) =>
        new(name, NpgsqlDbType.Array | elementType) { Value = values };
}
