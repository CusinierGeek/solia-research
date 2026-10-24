namespace Solia.Dashboard.Components.State;

/// <summary>The states of <see cref="AsyncContent{TItem}"/>'s load cycle.</summary>
public enum AsyncLoadState
{
    Loading,
    Empty,
    Error,
    Content,
}
