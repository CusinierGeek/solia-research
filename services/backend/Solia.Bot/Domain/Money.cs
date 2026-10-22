namespace Solia.Bot.Domain;

/// <summary>
/// Balance-unit precision. Money is held at <see cref="Decimals"/> places (SOL's lamport precision) so
/// sizes and P&amp;L stay short enough that <c>size · price</c> keeps every digit within
/// <see cref="decimal"/>'s 28–29 significant digits — no rounding residue piles up in the balance.
/// </summary>
public static class Money
{
    /// <summary>Decimal places money is kept at (1 lamport = 1e-9 SOL).</summary>
    public const int Decimals = 9;

    /// <summary>Rounds toward zero — for sizes, so a position never commits more than was computed.</summary>
    public static decimal Truncate(decimal amount) => Math.Round(amount, Decimals, MidpointRounding.ToZero);

    /// <summary>Rounds to nearest (banker's) — for realized P&amp;L.</summary>
    public static decimal Round(decimal amount) => Math.Round(amount, Decimals, MidpointRounding.ToEven);
}
