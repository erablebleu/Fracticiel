using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

public static partial class MultiSampling
{
    public static uint[] GPUInvoke(uint[] data, int w, int h, int samplingRate)
    {
        uint[] result = new uint[w * h];
        CudaException.Assert(GPUInvoke(result, data, w, h, samplingRate));
        return result;
    }

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "multisampling")]
    private static partial int GPUInvoke(uint[] result, uint[] data, int w, int h, int samplingRate);
}