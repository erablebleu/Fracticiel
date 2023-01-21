using System.Drawing;

namespace Fracticiel.Common.Coloring;

public interface IColorizer
{
    Bitmap GetBitmap(int[] data, int width, int height);
}