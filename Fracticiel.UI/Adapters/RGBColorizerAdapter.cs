using Fracticiel.UI.MVVM;

namespace Fracticiel.UI.Adapters;

public class RGBColorizerAdapter : ColorizerAdapter
{
    private GradientAdapter _gradient = new();

    public GradientAdapter Gradient { get => _gradient; set => Set(ref _gradient, value); }
}