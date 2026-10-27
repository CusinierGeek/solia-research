namespace Solia.Dashboard.Components.State;

/// <summary>The states of <see cref="AsyncContent{TItem}"/>'s load cycle.</summary>
public enum AsyncLoadState
{
    Loading,
    Empty,
    Error,

    /// <summary>The backend rejected the request as invalid (e.g. a filter out of range): retrying won't help.</summary>
    Invalid,
    Content,
}
