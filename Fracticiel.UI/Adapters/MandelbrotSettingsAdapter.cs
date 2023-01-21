using Fracticiel.UI.MVVM;

namespace Fracticiel.UI.Adapters;

public class MandelbrotSettingsAdapter : AdapterBase
{
    private int _loopCount = 10000;
    private double _magnitude = 2;

    public int LoopCount { get => _loopCount; set => Set(ref _loopCount, value); }
    public double Magnitude { get => _magnitude; set => Set(ref _magnitude, value); }
}