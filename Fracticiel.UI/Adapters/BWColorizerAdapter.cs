using System.ComponentModel;

namespace Fracticiel.UI.Adapters;

[Description("BW")]
public class BWColorizerAdapter : ColorizerAdapter
{
    private double _min;
    private double _max = 1;

    public double Min { get => _min; set => Set(ref _min, value); }
    public double Max { get => _max; set => Set(ref _max, value); }
}