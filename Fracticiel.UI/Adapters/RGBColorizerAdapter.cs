using System.ComponentModel;

namespace Fracticiel.UI.Adapters;

[Description("RGB")]
public class RGBColorizerAdapter : ColorizerAdapter
{
    private GradientAdapter _gradient = new();

    public GradientAdapter Gradient { get => _gradient; set => Set(ref _gradient, value); }
}