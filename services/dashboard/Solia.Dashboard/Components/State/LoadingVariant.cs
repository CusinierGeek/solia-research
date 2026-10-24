namespace Solia.Dashboard.Components.State;

/// <summary>How <see cref="LoadingState"/> draws itself.</summary>
public enum LoadingVariant
{
    /// <summary>Grey placeholder rows — for lists and tables.</summary>
    Skeleton,

    /// <summary>A centred spinner — for small widgets.</summary>
    Spinner,
}
