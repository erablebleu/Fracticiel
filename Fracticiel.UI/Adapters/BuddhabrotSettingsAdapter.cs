namespace Fracticiel.UI.Adapters;

public class BuddhabrotSettingsAdapter : MandelbrotSettingsAdapter
{
    private int _startPointCount = 100;

    public BuddhabrotSettingsAdapter()
    {
        Magnitude = 10;
    }

    public int StartPointCount { get => _startPointCount; set => Set(ref _startPointCount, value); }
}