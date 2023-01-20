using Fracticiel.Calculator;
using Fracticiel.UI.MVVM;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Policy;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace Fracticiel.UI.ViewModels;

public class MainViewModel : ViewModelBase
{
    private ICommand? _exportCommand;
    private ICommand? _drawCommand;
    private int _height = 1000;
    private int _width = 1000;
    private int _multiSampling = 2;
    //private double _widthRes = 0.074942;
    //private double _x = -1.41294724;
    //private double _y = -0.037471;
    private double _widthRes = 4;
    private double _x = -2;
    private double _y = -2;
    private BitmapSource? _bitmap;
    private double _progress;
    private bool _useBlockCalcultation = true;
    private int _blockCalculationSize = 500;
    private int _mode = 0;

    private void Draw()
    {
        DataBlock dataBlock = new()
        {
            Width = Width * MultiSampling,
            Height = Height * MultiSampling,
            X = X,
            Y = Y,
            Resolution = _widthRes / Width / MultiSampling,
        };
        Mandelbrot.Settings mSet = new()
        {
            LoopCount = 10000,
            Magnitude = 2.0,
        };
        Buddhabrot.Settings bSet = new()
        {
            LoopCount = 10000,
            Magnitude = 10,
            MaxValue = 50000,
        };
        Julia.Settings jSet = new()
        {
            LoopCount = 2000,
            Magnitude = 2,
            CX = 0.285,
            CY = 0.01,
        };
        Func<DataBlock, uint[]> builder = Mode switch
        {
            0 => db => Fracticiel.Calculator.Mandelbrot.GPUInvoke(db, mSet),
            1 => db => Fracticiel.Calculator.Buddhabrot.GPUInvoke(db, bSet),
            2 => db => Fracticiel.Calculator.Julia.GPUInvoke(db, jSet),
            _ => throw new NotImplementedException()
        };

        Draw(dataBlock, builder, UseBlockCalculation, BlockCalculationSize);
    }

    private void Draw(DataBlock dataBlock, Func<DataBlock, uint[]> builder, bool useBlocks, int blockSize)
    {
        Stopwatch sw = new();
        uint[] data;

        if(useBlocks)
        {
            blockSize = blockSize / MultiSampling * MultiSampling; // must be a multiple of MultiSampling

            (DataBlock, int X, int Y)[] blocks = Split(dataBlock, blockSize).ToArray();
            byte[,][] grid = new byte[blocks.Max(b => b.X) + 1, blocks.Max(b => b.Y) + 1][];
            byte[] bwData = new byte[Width * Height];

            // TODO: test parrallelization
            for (int i = 0; i < blocks.Length; i++)
            {
                (DataBlock block, int x, int y) = blocks[i];
                sw.Restart();

                uint[] blockData = builder.Invoke(block);

                if (MultiSampling > 1)
                    blockData = Calculator.MultiSampling.GPUInvoke(blockData, block.Width / MultiSampling, block.Height / MultiSampling, MultiSampling);

                byte[] result = Colorizer.BW(blockData, 0, 255);

                sw.Stop();

                // Add data to final array
                for (int j = 0; j < block.Height / MultiSampling; j++)
                    Array.Copy(result,
                               j * block.Width / MultiSampling,
                               bwData,
                               Width * (y * blockSize / MultiSampling + j) + x * blockSize / MultiSampling,
                               block.Width / MultiSampling);

                Debug.WriteLine($"bloc {i + 1}/{blocks.Length} calculated in {sw.ElapsedMilliseconds} ms");
                Progress = (double)(i + 1) / blocks.Length;
                Bitmap = GetBitmap(bwData, Width, Height);
            }
        }
        else
        {
            data = builder.Invoke(dataBlock);
            if (MultiSampling > 1)
                data = Fracticiel.Calculator.MultiSampling.GPUInvoke(data, Width, Height, MultiSampling);
            Bitmap = GetBitmap(Fracticiel.Calculator.Colorizer.BW(data, 0, 255), Width, Height);
        }
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

    private static BitmapSource GetBitmap(byte[] data, int width, int height)
    {
        PixelFormat pf = PixelFormats.Gray8;
        int rawStride = (width * pf.BitsPerPixel + 7) / 8;
        BitmapSource result = new TransformedBitmap(BitmapSource.Create(width, height, 96, 96, pf, null, data, rawStride), new RotateTransform(90));
        result.Freeze();
        // Create a BitmapSource.
        return result;

    }

    public ICommand ExportCommand => _exportCommand ??= new RelayCommand(OnExportCommand);
    public ICommand DrawCommand => _drawCommand ??= new RelayCommand(OnDrawCommand);
    public int Height { get => _height; set => Set(ref _height, value); }
    public int MultiSampling { get => _multiSampling; set => Set(ref _multiSampling, value); }
    public int Width { get => _width; set => Set(ref _width, value); }
    public double WidthRes { get => _widthRes; set => Set(ref _widthRes, value); }
    public double X { get => _x; set => Set(ref _x, value); }
    public double Y { get => _y; set => Set(ref _y, value); }
    public double Progress { get => _progress; set => Set(ref _progress, value); }
    public BitmapSource? Bitmap { get => _bitmap; set => Set(ref _bitmap, value); }
    public bool UseBlockCalculation { get => _useBlockCalcultation; set => Set(ref _useBlockCalcultation, value); }
    public int BlockCalculationSize { get => _blockCalculationSize; set => Set(ref _blockCalculationSize, value); }
    public int Mode { get => _mode; set => Set(ref _mode, value); }

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

    private void OnDrawCommand()
    {
        Task.Run(Draw);
    }
}