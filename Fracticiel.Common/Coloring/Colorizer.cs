using System.Drawing;

namespace Fracticiel.Common.Coloring;

public abstract class Colorizer<T> : IColorizer
{
    public abstract T[] Colorize(int[] data);

    public virtual Bitmap GetBitmap(int[] data, int width, int height) => GenerateBitmap(Colorize(data), width, height);

    protected abstract Bitmap GenerateBitmap(T[] data, int width, int height);
}