using System.Drawing;

namespace Fracticiel.Common.Coloring;

public class Gradient
{
    public List<GradientStop> Stops = new();

    public Color GetColor(double position)
    {
        GradientStop[] stops = Stops.OrderBy(s => s.Position).ToArray();

        if (stops.Length == 0)
            return Color.Black;

        if (position < stops[0].Position)
            return stops[0].Color;

        for (int i = 0; i < stops.Length - 1; i++)
        {
            if (position >= stops[i + 1].Position)
                continue;
            Color src = stops[i].Color;
            Color dst = stops[i + 1].Color;
            double d = stops[i + 1].Position - stops[i].Position;
            double x = position - stops[i].Position;

            return Color.FromArgb(
                src.A + (int)((dst.A - src.A) * x / d),
                src.R + (int)((dst.R - src.R) * x / d),
                src.G + (int)((dst.G - src.G) * x / d),
                src.B + (int)((dst.B - src.B) * x / d));
        }

        return stops.Last().Color;
    }
}