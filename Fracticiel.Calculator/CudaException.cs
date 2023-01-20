using System.Runtime.InteropServices;

namespace Fracticiel.Calculator;

public partial class CudaException : Exception
{
    private CudaException(int code) : base(GetMessage())
    {
        Name = GetName();
        FileName = GetFileName();
        FileLine = GetFileLine();
        Code = code;
    }

    public int FileLine { get; private set; }
    public string FileName { get; private set; }
    public string Name { get; private set; }
    public int Code { get; private set; }

    public static void Assert (int code)
    {
        if (code == 0) return;
        throw new CudaException(code);
    }

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "getLastErrorName")]
    private static partial int GetFileLine();

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "getLastErrorFileName", StringMarshalling = StringMarshalling.Utf16)]
    private static partial string GetFileName();

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "getLastErrorString", StringMarshalling = StringMarshalling.Utf16)]
    private static partial string GetMessage();

    [LibraryImport("Fracticiel.Cuda.Calculator.dll", EntryPoint = "getLastErrorName", StringMarshalling = StringMarshalling.Utf16)]
    private static partial string GetName();
}