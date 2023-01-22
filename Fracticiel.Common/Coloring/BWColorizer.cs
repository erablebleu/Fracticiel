using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Fracticiel.Common.Coloring;

public class BWColorizer : Colorizer<byte>
{
    public double Max;
    public double Min;

    public override byte[] Colorize(int[] data)
    {
        int max = (int)(Max * data.Max());
        int min = (int)(Min * data.Max());
        int d = max - min;
        byte[] result = new byte[data.Length];

        for (int i = 0; i < data.Length; i++)
            result[i] = (byte)(data[i] <= min ? 0 : data[i] >= max ? 255 : ((data[i] - min) * 255 / d));

        return result;
    }

    protected override Bitmap GenerateBitmap(byte[] data, int width, int height)
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