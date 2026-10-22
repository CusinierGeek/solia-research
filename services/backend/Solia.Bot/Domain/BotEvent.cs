namespace Solia.Bot.Domain;

/// <summary>
/// The single stream the bot consumes: either a new <see cref="Prediction"/> (a signal to possibly
/// act on) or a <see cref="Price"/> (a mark to update open positions). A closed union — exactly these
/// two cases — so consumers pattern-match exhaustively. Closed by the private constructor (nested types
/// only) and by <see cref="CloseUnion"/>: a record also gets a protected copy constructor that would let
/// outside code derive, but no type outside this assembly can implement a <c>private protected</c>
/// abstract member, so such a subtype can't be instantiated. Every event carries a <see cref="Ts"/> and the
/// <see cref="Mint"/> it concerns, letting the engine order and route events without unwrapping them.
/// </summary>
public abstract record BotEvent
{
    private BotEvent() { }

    /// <summary>Not implementable outside this assembly — keeps the union closed (see the type summary).</summary>
    private protected abstract void CloseUnion();

    /// <summary>Logical time of the event, used to order the stream.</summary>
    public abstract DateTimeOffset Ts { get; }

    /// <summary>Token the event concerns.</summary>
    public abstract string Mint { get; }

    /// <summary>A model prediction arrived for a token.</summary>
    public sealed record Prediction(ModelPrediction Value) : BotEvent
    {
        public override DateTimeOffset Ts => Value.Ts;
        public override string Mint => Value.Mint;
        private protected override void CloseUnion() { }
    }

    /// <summary>A new price was observed for a token.</summary>
    public sealed record Price(PriceTick Value) : BotEvent
    {
        public override DateTimeOffset Ts => Value.Ts;
        public override string Mint => Value.Mint;
        private protected override void CloseUnion() { }
    }
}
