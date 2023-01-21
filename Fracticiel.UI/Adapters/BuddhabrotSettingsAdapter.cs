using Fracticiel.UI.MVVM;

namespace Fracticiel.UI.Adapters;

public class BuddhabrotSettingsAdapter : MandelbrotSettingsAdapter
{
    private int _maxValue = 50000;

    public BuddhabrotSettingsAdapter()
    {
        Magnitude = 10;
    }

    public int MaxValue { get => _maxValue; set => Set(ref _maxValue, value); }
}