using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Fracticiel.Common.Coloring;

public class RGBColorizer : Colorizer<int>
{
    public Gradient Gradient = new()
    {
        Stops = new List<GradientStop>()
        {
            new GradientStop(0, Color.Black),
            new GradientStop(0.1, Color.Blue),
            new GradientStop(0.2, Color.Red),
            new GradientStop(0.4, Color.OrangeRed),
            new GradientStop(0.6, Color.Orange),
            new GradientStop(0.8, Color.Yellow),
            new GradientStop(1, Color.White),
        }
    };

    public override int[] Colorize(uint[] data)
    {
        uint min = data.Min();
        uint max = data.Max();
        return data.Select(i => Gradient.GetColor((double)(i - min) / (max - min)).ToArgb()).ToArray();
    }

    public override Bitmap GetBitmap(int[] data, int width, int height)
    {
        Bitmap result = new(width, height, PixelFormat.Format32bppArgb);
        BitmapData bmpData = result.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, result.PixelFormat);
        Marshal.Copy(data, 0, bmpData.Scan0, data.Length);
        result.UnlockBits(bmpData);
        result.RotateFlip(RotateFlipType.Rotate90FlipNone);

        return result;
    }
}