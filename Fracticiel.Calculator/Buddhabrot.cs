using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

public static partial class Buddhabrot
{
    private static Random Random = new(Guid.NewGuid().GetHashCode());

    public static uint[] GPUInvoke(DataBlock block, Settings settings)
    {
        uint[] result = new uint[block.Width * block.Height];
        double[] randoms = new double[2 * block.Width * block.Height];
        for(int i = 0; i < block.Width * block.Height; i++)
        {
            randoms[2 * i] = block.X + block.Width * block.Resolution * Random.NextDouble();
            randoms[2 * i + 1] = block.Y + block.Height * block.Resolution * Random.NextDouble();

            //randoms[2 * i] = -1.5 + 3 * Random.NextDouble();
            //randoms[2 * i + 1] = - 1.5 + 3 * Random.NextDouble();
        }
            
        CudaException.Assert(GPUInvoke(result, block, settings, randoms));
        return result;
    }

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "buddhabrot")]
    private static partial int GPUInvoke(uint[] result, DataBlock block, Settings settings, double[] randoms);

    [StructLayout(LayoutKind.Sequential)]
    public struct Settings
    {
        public int LoopCount;
        public double Magnitude;
        public int MaxValue;
    }
}