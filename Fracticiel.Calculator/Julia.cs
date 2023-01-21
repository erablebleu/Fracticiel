using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

public static partial class Julia
{
    public static int[] Get(DataBlock block, Settings settings)
    {
        int[] result = new int[block.Width * block.Height];
        CudaException.Assert(GPUInvoke(result, block, settings));
        return result;
    }

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "julia")]
    private static partial int GPUInvoke(int[] result, DataBlock block, Settings settings);

    [StructLayout(LayoutKind.Sequential)]
    public struct Settings
    {
        public int LoopCount;
        public double Magnitude;
        public double CX;
        public double CY;
    }
}