using MudBlazor;

namespace Solia.Dashboard.Components.Layout;

/// <summary>The dashboard's MudBlazor theme (dark by default, light available from the app bar).</summary>
internal static class SoliaTheme
{
    public static MudTheme Theme { get; } = new()
    {
        PaletteLight = new PaletteLight
        {
            Primary = "#5b4bdb",
            Secondary = "#00a884",
            AppbarBackground = "#1f1b3a",
        },
        PaletteDark = new PaletteDark
        {
            Primary = "#8b7cf6",
            Secondary = "#2dd4a7",
            Background = "#121218",
            Surface = "#1b1b24",
            DrawerBackground = "#16161e",
            AppbarBackground = "#16161e",
        },
    };
}
