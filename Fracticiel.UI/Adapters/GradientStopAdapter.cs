using Fracticiel.UI.MVVM;
using System.Drawing;

namespace Fracticiel.UI.Adapters;

public class GradientStopAdapter : AdapterBase
{
    private Color _color;
    private double _position;
    public Color Color { get => _color; set => Set(ref _color, value); }
    public double Position { get => _position; set => Set(ref _position, value); }
}