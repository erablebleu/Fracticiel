using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

public static partial class Buddhabrot
{
    private static Random Random = new(Guid.NewGuid().GetHashCode());

    public static uint[] GPUInvoke(DataBlock block, Settings settings)
    {
        uint[] result = new uint[block.Width * block.Height];

        CudaException.Assert(GPUInvoke(result, block, settings));
        return result;
    }

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "buddhabrot")]
    private static partial int GPUInvoke(uint[] result, DataBlock block, Settings settings);

    [StructLayout(LayoutKind.Sequential)]
    public struct Settings
    {
        public int LoopCount;
        public double Magnitude;
        public int MaxValue;
    }
}