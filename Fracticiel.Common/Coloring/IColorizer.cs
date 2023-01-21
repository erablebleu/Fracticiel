using System.Drawing;

namespace Fracticiel.Common.Coloring;

public interface IColorizer
{
    Bitmap GetBitmap(uint[] data, int width, int height);
}