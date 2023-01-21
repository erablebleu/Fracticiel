using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Fracticiel.Common.Coloring;

public class BWColorizer : Colorizer<byte>
{
    public override byte[] Colorize(uint[] data)
    {
        uint min = data.Min();
        uint max = data.Max();
        return data.Select(i => 255 * (i - min) / (max - min)).Select(i => (byte)i).ToArray();
    }

    public override Bitmap GetBitmap(byte[] data, int width, int height)
    {
        Bitmap result = new(width, height, PixelFormat.Format8bppIndexed);

        ColorPalette ncp = result.Palette;
        for (int i = 0; i < 256; i++)
            ncp.Entries[i] = Color.FromArgb(255, i, i, i);
        result.Palette = ncp;

        BitmapData bmpData = result.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, result.PixelFormat);
        Marshal.Copy(data, 0, bmpData.Scan0, bmpData.Stride * result.Height);
        result.UnlockBits(bmpData);
        result.RotateFlip(RotateFlipType.Rotate90FlipNone);

        return result;
    }
}