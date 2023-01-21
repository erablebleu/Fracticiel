using Fracticiel.Calculator;
using Fracticiel.UI.MVVM;

namespace Fracticiel.UI.Adapters;

public class CalculationSettingsAdapter : AdapterBase
{
    private int _blockCalculationSize = 500;
    private int _height = 500;
    private int _multiSampling = 4;
    private bool _useBlockCalcultation = false;
    private int _width = 500;

    //private double _widthRes = 0.074942;
    //private double _x = -1.41294724;
    //private double _y = -0.037471;
    private double _widthRes = 4;

    private double _x = -2;
    private double _y = -2;

    public int BlockCalculationSize { get => _blockCalculationSize; set => Set(ref _blockCalculationSize, value); }
    public int Height { get => _height; set => Set(ref _height, value); }
    public int MultiSampling { get => _multiSampling; set => Set(ref _multiSampling, value); }
    public bool UseBlockCalculation { get => _useBlockCalcultation; set => Set(ref _useBlockCalcultation, value); }

    public int Width { get => _width; set => Set(ref _width, value); }

    public double WidthRes { get => _widthRes; set => Set(ref _widthRes, value); }

    public double X { get => _x; set => Set(ref _x, value); }

    public double Y { get => _y; set => Set(ref _y, value); }

    public DataBlock GetDataBlock() => new()
    {
        Width = Width,
        Height = Height,
        X = X,
        Y = Y,
        Resolution = _widthRes / Width,
        MultiSampling = MultiSampling,
    };
}