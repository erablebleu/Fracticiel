using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

public static partial class Colorizer
{
    public static byte[] BW(uint[] data, uint min, uint max)
    {
        byte[] result = new byte[data.Length];        
        CudaException.Assert(BW(result, data, data.Length, min, max));
        return result;
    }

    public static uint[] RGB(uint[] data, uint min, uint max)
    {
        uint[] result = new uint[data.Length];
        CudaException.Assert(RGB(result, data, data.Length, min, max));
        return result;
    }

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "colorizeBW")]
    private static partial int BW(byte[] result, uint[] data, int sz, uint min, uint max);

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "colorizeRGB")]
    private static partial int RGB(uint[] result, uint[] data, int sz, uint min, uint max);
}