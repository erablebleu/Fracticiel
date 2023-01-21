using Fracticiel.UI.MVVM;
using System.Collections.ObjectModel;
using System.Drawing;

namespace Fracticiel.UI.Adapters;

public class GradientAdapter : AdapterBase
{
    private ObservableCollection<GradientStopAdapter> _stops = new()
    {
            new GradientStopAdapter() { Position=0, Color= Color.Black },
            new GradientStopAdapter() { Position=0.1, Color= Color.Blue },
            new GradientStopAdapter() { Position=0.2, Color= Color.Red },
            new GradientStopAdapter() { Position=0.4, Color= Color.OrangeRed },
            new GradientStopAdapter() { Position=0.6, Color= Color.Orange },
            new GradientStopAdapter() { Position=0.8, Color= Color.Yellow },
            new GradientStopAdapter() { Position=1, Color= Color.White },
    };

    public ObservableCollection<GradientStopAdapter> Stops { get => _stops; set => Set(ref _stops, value); }
}