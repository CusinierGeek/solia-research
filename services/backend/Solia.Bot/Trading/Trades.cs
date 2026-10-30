using Solia.Bot.Domain;
using Solia.Bot.Policy;

namespace Solia.Bot.Trading;

/// <summary>
/// Everything fixed when a paper position is opened — what <see cref="ITradeStore.OpenAsync"/>
/// records (the <c>trades</c> row's entry side). Money/price are <see cref="decimal"/>.
/// </summary>
/// <param name="Mint">Token held.</param>
/// <param name="OpenedAt">Event time of the fill: the entry prediction's timestamp, or the event clock if it arrived late.</param>
/// <param name="EntryPrice">
/// All-in price per token the order filled at, in SOL: with realistic fills the mark plus price impact and the
/// entry fee (<see cref="Fills.BuyPrice"/>), else the mark.
/// </param>
/// <param name="Size">Capital committed, in SOL, locked from the balance until close.</param>
/// <param name="Levels">Dynamic TP/SL fixed at entry, from the mark (not the all-in price).</param>
/// <param name="Prediction">The prediction that opened the trade — confidence, model version and the features snapshot.</param>
public sealed record TradeOpening(
    string Mint,
    DateTimeOffset OpenedAt,
    decimal EntryPrice,
    decimal Size,
    ExitLevels Levels,
    ModelPrediction Prediction)
{
    /// <summary>Entry strategy (<c>trades.strategy</c>) — the bot only opens long on a buy signal.</summary>
    public TradeSignal Strategy => TradeSignal.Buy;

    /// <summary>
    /// Value of the position at <paramref name="price"/>: <c>Size · price / EntryPrice</c>, multiplied
    /// before dividing. A price equal to the entry returns exactly <see cref="Size"/> — break-even is 0 by
    /// construction, not by luck of <see cref="decimal"/> rounding on long operands.
    /// </summary>
    public decimal ValueAt(decimal price) => price == EntryPrice ? Size : Size * price / EntryPrice;
}

/// <summary>A position currently held, with the id the <see cref="ITradeStore"/> assigned it.</summary>
public sealed record OpenPosition(long TradeId, TradeOpening Entry, DateTimeOffset Deadline)
{
    public string Mint => Entry.Mint;
}

/// <summary>
/// A closed paper trade — what <see cref="ITradeStore.CloseAsync"/> records (the <c>trades</c> row's
/// exit side). <see cref="Pnl"/> is realized against the committed size.
/// </summary>
/// <param name="TradeId">Id returned when the trade was opened.</param>
/// <param name="Entry">The opening the trade closes.</param>
/// <param name="ClosedAt">Event time of the exit.</param>
/// <param name="ExitPrice">Price per token the exit filled at, in SOL (with realistic fills: after impact and the exit fee).</param>
/// <param name="Reason">Why it closed.</param>
/// <param name="Pnl">Realized profit/loss in SOL (fees included): proceeds − size, at <see cref="Money.Decimals"/> places.</param>
/// <param name="PnlPct">Realized return on the committed size (0.25 = +25%).</param>
/// <param name="DurationSec">Whole seconds the position was held.</param>
public sealed record ClosedTrade(
    long TradeId,
    TradeOpening Entry,
    DateTimeOffset ClosedAt,
    decimal ExitPrice,
    ExitReason Reason,
    decimal Pnl,
    double PnlPct,
    int DurationSec)
{
    /// <summary>Settles <paramref name="position"/> at <paramref name="exitPrice"/> — the P&amp;L math, in one place.</summary>
    public static ClosedTrade Settle(OpenPosition position, DateTimeOffset closedAt, decimal exitPrice, ExitReason reason)
    {
        ArgumentNullException.ThrowIfNull(position);

        TradeOpening entry = position.Entry;
        // Rounded to money precision so the balance it feeds stays short (see Money).
        decimal pnl = Money.Round(entry.ValueAt(exitPrice) - entry.Size);
        return new ClosedTrade(
            position.TradeId,
            entry,
            closedAt,
            exitPrice,
            reason,
            pnl,
            (double)(pnl / entry.Size),
            (int)(closedAt - entry.OpenedAt).TotalSeconds);
    }

    /// <summary>What the close returns to the balance: the position's value at the exit price.</summary>
    public decimal Proceeds => Entry.Size + Pnl;
}
