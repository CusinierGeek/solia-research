using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Trading;

namespace Solia.Bot.Policy;

/// <summary>
/// The entry gate: decides whether a prediction should open a new position, given the portfolio's
/// current capacity, and how much to commit. Pure and deterministic — no I/O, no clock. Gates are
/// checked in a fixed order and the <em>first</em> failure is reported, so the reason is stable and
/// testable:
/// <list type="number">
///   <item>the prediction is well-formed (<see cref="PredictionValidation"/>);</item>
///   <item>confidence ≥ <see cref="BotOptions.MinConfidence"/> (<see cref="SignalClassifier.IsConfident"/>);</item>
///   <item>the classified signal is <see cref="TradeSignal.Buy"/>;</item>
///   <item>below <see cref="BotOptions.MaxConcurrentPositions"/>;</item>
///   <item>no open position already holds this mint;</item>
///   <item>a positive market price exists to fill against;</item>
///   <item>with <see cref="BotOptions.RealisticFills"/>, its pool clears the liquidity gate
///   (<see cref="Fills.IsTradeable"/>: known liquidity, at least the venue's floor);</item>
///   <item>balance &gt; 0 and ≥ <see cref="BotOptions.MinPositionSize"/>;</item>
///   <item>the bet has a positive edge (<see cref="PositionSizer.Fraction"/> &gt; 0);</item>
///   <item>the sized amount — capped by the pool with realistic fills — clears the
///   <see cref="BotOptions.MinPositionSize"/> dust floor.</item>
/// </list>
/// An accept therefore always carries a positive <see cref="EntryDecision.Size"/>, sized by
/// <see cref="PositionSizer"/> from the same <see cref="PortfolioState"/> the gates checked.
/// Confidence is gated explicitly — ahead of the signal — so a precise
/// <see cref="EntryRejectReason.LowConfidence"/> is reported rather than being folded into the
/// classifier's <see cref="TradeSignal.Hold"/>.
/// </summary>
public static class EntryRule
{
    public static EntryDecision Evaluate(ModelPrediction prediction, PortfolioState portfolio, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(options);

        if (!PredictionValidation.IsWellFormed(prediction))
        {
            return EntryDecision.Reject(EntryRejectReason.InvalidPrediction);
        }

        if (!SignalClassifier.IsConfident(prediction, options))
        {
            return EntryDecision.Reject(EntryRejectReason.LowConfidence);
        }

        if (SignalClassifier.Classify(prediction, options) != TradeSignal.Buy)
        {
            return EntryDecision.Reject(EntryRejectReason.NotABuySignal);
        }

        if (portfolio.OpenPositionCount >= options.MaxConcurrentPositions)
        {
            return EntryDecision.Reject(EntryRejectReason.AtMaxConcurrentPositions);
        }

        if (portfolio.HoldsMint)
        {
            return EntryDecision.Reject(EntryRejectReason.AlreadyHoldingMint);
        }

        if (portfolio.MarketPrice is not > 0m)
        {
            return EntryDecision.Reject(EntryRejectReason.NoMarketPrice);
        }

        if (options.RealisticFills && !Fills.IsTradeable(portfolio.Pool, options))
        {
            return EntryDecision.Reject(EntryRejectReason.InsufficientLiquidity);
        }

        if (portfolio.Balance <= 0m || portfolio.Balance < options.MinPositionSize)
        {
            return EntryDecision.Reject(EntryRejectReason.InsufficientBalance);
        }

        if (PositionSizer.Fraction(prediction, options) <= 0.0)
        {
            return EntryDecision.Reject(EntryRejectReason.NoPositiveEdge);
        }

        decimal size = PositionSizer.Size(prediction, portfolio, options);
        return size > 0m
            ? EntryDecision.Accept(size)
            : EntryDecision.Reject(EntryRejectReason.BelowMinPositionSize);
    }
}
