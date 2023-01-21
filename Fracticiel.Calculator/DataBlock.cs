using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

[StructLayout(LayoutKind.Sequential)]
public struct DataBlock
{
    public int Width;
    public int Height;
    public double X;
    public double Y;
    public double Resolution;
    public int MultiSampling;
}