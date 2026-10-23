using Npgsql;
using NpgsqlTypes;
using Solia.Bot.Domain;
using static Solia.Shared.Data.NpgsqlParameters;

namespace Solia.Api.Bot;

/// <summary>
/// Persists the predictions the bot's feed scores to the <c>predictions</c> table (raw SQL, like every
/// write — EF stays read-only), so a trade opened on one carries a real <c>prediction_id</c>. Predictions
/// only: the policy columns (<c>strategy</c>, sizing, TP/SL) stay null — decisions are the bot's, recorded
/// on the trade. <c>model_id</c> is looked up by the champion version (null when the registry doesn't know it).
/// </summary>
public sealed class PostgresPredictionStore(NpgsqlDataSource dataSource)
{
    private const string InsertSql =
        "INSERT INTO predictions (mint, model_id, ts, horizon, pump_prob, rug_prob, confidence, feature_id) " +
        "VALUES (@mint, (SELECT id FROM models WHERE version = @version), @ts, @horizon, @pump, @rug, @confidence, @feature) " +
        "RETURNING id";

    /// <summary>Inserts <paramref name="prediction"/> (computed from feature row <paramref name="featureId"/>) and returns its id.</summary>
    public async Task<long> InsertAsync(ModelPrediction prediction, long? featureId, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prediction);

        await using NpgsqlCommand cmd = dataSource.CreateCommand(InsertSql);
        cmd.Parameters.Add(Param("mint", NpgsqlDbType.Text, prediction.Mint));
        cmd.Parameters.Add(Param("version", NpgsqlDbType.Text, prediction.ModelVersion));
        cmd.Parameters.Add(Param("ts", NpgsqlDbType.TimestampTz, prediction.Ts));
        cmd.Parameters.Add(Param("horizon", NpgsqlDbType.Text, prediction.Horizon));
        cmd.Parameters.Add(Param("pump", NpgsqlDbType.Double, prediction.PumpProb));
        cmd.Parameters.Add(Param("rug", NpgsqlDbType.Double, prediction.RugProb));
        cmd.Parameters.Add(Param("confidence", NpgsqlDbType.Double, prediction.Confidence));
        cmd.Parameters.Add(Param("feature", NpgsqlDbType.Bigint, featureId));
        return (long)(await cmd.ExecuteScalarAsync(cancellationToken))!;
    }
}
