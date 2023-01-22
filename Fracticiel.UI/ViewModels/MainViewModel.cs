using Fracticiel.Calculator;
using Fracticiel.Common.Coloring;
using Fracticiel.UI.Adapters;
using Fracticiel.UI.MVVM;
using Fracticiel.UI.Tools;
using Fracticiel.UI.Tools.Serialization;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Fracticiel.UI.ViewModels;

public class AdapterSaver<T> where T : AdapterBase, new()
{
    public readonly T Adapter;
    private readonly string _filePath;

    public AdapterSaver(string filePath)
    {
        _filePath = filePath;
        Adapter = File.Exists(_filePath) ? JsonSerializer.Deserialize<T>(File.ReadAllText(_filePath)) ?? new T() : new T();
        Adapter.PropertyChanged += (sender, args) => File.WriteAllText(_filePath, JsonSerializer.Serialize(Adapter));
    }
}

public class MainViewModel : ViewModelBase
{
    private BitmapSource? _bitmap;
    private BuddhabrotSettingsAdapter _buddhabrotSettings = new();
    private ICommand? _calculateCommand;
    private CalculationSettingsAdapter _calculationSettings = new();
    private int _coloringMode = 0;
    private ICommand? _colorizeCommand;

    private ObservableCollection<ColorizerAdapter> _colorizers = new()
    {
        new BWColorizerAdapter(),
        new RGBColorizerAdapter(),
    };

    private int[] _data;
    private DataBlock _dataBlock;
    private ICommand? _exportCommand;
    private JuliaSettingsAdapter _juliaSettings = new();
    private MandelbrotSettingsAdapter _mandelbrotSettings = new();
    private int _mode = 1;
    private double _progress;
    private ColorizerAdapter? _selectedColorizer;

    public BitmapSource? Bitmap { get => _bitmap; set => Set(ref _bitmap, value); }
    public BuddhabrotSettingsAdapter BuddhabrotSettings { get => _buddhabrotSettings; set => Set(ref _buddhabrotSettings, value); }
    public ICommand CalculateCommand => _calculateCommand ??= new RelayCommand(OnCalculateCommand);
    public CalculationSettingsAdapter CalculationSettings { get => _calculationSettings; set => Set(ref _calculationSettings, value); }
    public int ColoringMode { get => _coloringMode; set => Set(ref _coloringMode, value); }
    public ICommand ColorizeCommand => _colorizeCommand ??= new RelayCommand(OnColorizeCommand);
    public ObservableCollection<ColorizerAdapter> Colorizers { get => _colorizers; set => Set(ref _colorizers, value); }
    public ICommand ExportCommand => _exportCommand ??= new RelayCommand(OnExportCommand);
    public JuliaSettingsAdapter JuliaSettings { get => _juliaSettings; set => Set(ref _juliaSettings, value); }
    public MandelbrotSettingsAdapter MandelbrotSettings { get => _mandelbrotSettings; set => Set(ref _mandelbrotSettings, value); }
    public int Mode { get => _mode; set => Set(ref _mode, value); }
    public double Progress { get => _progress; set => Set(ref _progress, value); }
    public ColorizerAdapter? SelectedColorizer { get => _selectedColorizer; set => Set(ref _selectedColorizer, value); }

    public override void Load()
    {
        BuddhabrotSettings = new AdapterSaver<BuddhabrotSettingsAdapter>(@"buddhabrot.json").Adapter;
        MandelbrotSettings = new AdapterSaver<MandelbrotSettingsAdapter>(@"mandelbrot.json").Adapter;
        JuliaSettings = new AdapterSaver<JuliaSettingsAdapter>(@"julia.json").Adapter;
        CalculationSettings = new AdapterSaver<CalculationSettingsAdapter>(@"calculation.json").Adapter;

        SelectedColorizer = Colorizers.First();
    }

    private static IEnumerable<(DataBlock, int X, int Y)> Split(DataBlock dataBlock, int size)
    {
        int xCount = dataBlock.Width / size + (dataBlock.Width % size > 0 ? 1 : 0);
        int yCount = dataBlock.Height / size + (dataBlock.Height % size > 0 ? 1 : 0);

        for (int x = 0; x < xCount; x++)
        {
            for (int y = 0; y < yCount; y++)
            {
                yield return (new DataBlock
                {
                    Width = Math.Min(size, dataBlock.Width - x * size),
                    Height = Math.Min(size, dataBlock.Height - y * size),
                    X = dataBlock.X + x * size * dataBlock.Resolution,
                    Y = dataBlock.Y + y * size * dataBlock.Resolution,
                    Resolution = dataBlock.Resolution
                }, x, y);
            }
        }

        yield break;
    }

    private int[] Calculate(DataBlock dataBlock, Func<DataBlock, int[]> builder, bool useBlocks, int blockSize, int width, int height)
    {
        if(!useBlocks)
            return builder.Invoke(dataBlock);

        Stopwatch sw = new();
        int[] data = new int[width * height];

        (DataBlock, int X, int Y)[] blocks = Split(dataBlock, blockSize).ToArray();
        byte[,][] grid = new byte[blocks.Max(b => b.X) + 1, blocks.Max(b => b.Y) + 1][];

        // TODO: test parrallelization
        for (int i = 0; i < blocks.Length; i++)
        {
            (DataBlock block, int x, int y) = blocks[i];
            sw.Restart();

            int[] blockData = builder.Invoke(block);

            sw.Stop();

            // Add data to final array
            for (int j = 0; j < block.Height; j++)
                Array.Copy(blockData,
                            j * block.Width,
                            data,
                            width * (y * blockSize + j) + x * blockSize,
                            block.Width);

            Debug.WriteLine($"bloc {i + 1}/{blocks.Length} calculated in {sw.ElapsedMilliseconds} ms");
            Progress = (double)(i + 1) / blocks.Length;
        }

        return data;
    }

    private void Calculate()
    {
        _dataBlock = CalculationSettings.GetDataBlock();
        Func<DataBlock, int[]> builder = Mode switch
        {
            0 => db => Fracticiel.Calculator.Mandelbrot.Get(db, Mapper.Map<Mandelbrot.Settings>(MandelbrotSettings)),
            1 => db => Fracticiel.Calculator.Buddhabrot.Get(db, Mapper.Map<Buddhabrot.Settings>(BuddhabrotSettings)),
            2 => db => Fracticiel.Calculator.Julia.Get(db, Mapper.Map<Julia.Settings>(JuliaSettings)),
            _ => throw new NotImplementedException()
        };
        _data = Calculate(_dataBlock, builder,
            CalculationSettings.UseBlockCalculation,
            CalculationSettings.BlockCalculationSize,
            CalculationSettings.Width,
            CalculationSettings.Height);
    }

    private void Colorize()
    {
        IColorizer colorizer = SelectedColorizer switch
        {
            BWColorizerAdapter => Mapper.Map<BWColorizer>(SelectedColorizer),
            RGBColorizerAdapter => Mapper.Map<RGBColorizer>(SelectedColorizer),
            _ => throw new NotImplementedException()
        };
        Bitmap = colorizer.GetBitmap(_data, CalculationSettings.Width, CalculationSettings.Height).ToBitmapImage();
    }

    private void OnCalculateCommand()
    {
        Task.Run(Calculate);
    }

    private void OnColorizeCommand()
    {
        Task.Run(Colorize);
    }

    private void OnExportCommand()
    {
        Microsoft.Win32.SaveFileDialog sfd = new()
        {
            Filter = "Bitmap file|*.bmp|PNG|*.png|JPEG|*.jpg;*.jpeg;*.jpe;*.jfif",
        };
        if (!sfd.ShowDialog() == true) return;

        BitmapEncoder encoder = Path.GetExtension(sfd.FileName) switch
        {
            ".bmp" or ".png" => new PngBitmapEncoder(),
            ".jpg" or ".jpeg" or ".jpe" or ".jfif" => new JpegBitmapEncoder(),
            _ => throw new NotImplementedException(),
        };

        using var fileStream = new FileStream(sfd.FileName, FileMode.Create);
        encoder.Frames.Add(BitmapFrame.Create(Bitmap));
        encoder.Save(fileStream);
    }
}