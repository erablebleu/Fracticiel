using System.Drawing;

namespace Fracticiel.Common.Coloring;

public abstract class Colorizer<T> : IColorizer
{
    public abstract T[] Colorize(uint[] data);

    public virtual Bitmap GetBitmap(uint[] data, int width, int height) => GetBitmap(Colorize(data), width, height);

    public abstract Bitmap GetBitmap(T[] data, int width, int height);
}