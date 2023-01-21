namespace Fracticiel.UI.Adapters;

public class JuliaSettingsAdapter : MandelbrotSettingsAdapter
{
    private double _cx = 0.285;
    private double _cy = 0.01;

    public JuliaSettingsAdapter()
    {
        LoopCount = 2000;
    }

    public double CX { get => _cx; set => Set(ref _cx, value); }
    public double CY { get => _cy; set => Set(ref _cy, value); }
}