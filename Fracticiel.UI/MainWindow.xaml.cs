using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace Fracticiel.UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const int _h = 500;
        private const int _multisampling = 2;
        private const int _w = 500;
        private System.Windows.Point _downP;
        private bool _mouseIn;
        private double _res = 4.0 / _w;
        private double _x0 = -2;
        private double _y0 = -2;
        private uint[] arr = new uint[_w * _multisampling * _h * _multisampling];
        private byte[] colorsBW = new byte[_w * _h];
        private uint[] colorsRGB = new uint[_w * _h];
        private uint[] samples = new uint[_w * _h];
        private WriteableBitmap _bitmap;
        IntPtr _bitmapData;

        public MainWindow()
        {
            InitializeComponent();
            image.Width = _w;
            image.Height = _h;
            byte[] data = new byte[_h * _w];
            _bitmap = new WriteableBitmap(BitmapSource.Create(_w, _h, 96, 96, System.Windows.Media.PixelFormats.Gray8, null, data, _w * 1));
            _bitmapData = _bitmap.BackBuffer;
            image.Source = _bitmap;

            Draw();
        }

        public static BitmapSource BitmapFromDataBW(int w, int h, byte[] data)
        {
            return BitmapSource.Create(w, h, 96, 96, System.Windows.Media.PixelFormats.Gray8, null, data, w);
        }
        public static BitmapSource BitmapFromDataRGB(int w, int h, uint[] data)
        {
            return BitmapSource.Create(w, h, 96, 96, System.Windows.Media.PixelFormats.Bgra32, null, data, w * 4);
        }

        private void Draw()
        {
            var sw01 = new Stopwatch();
            var sw02 = new Stopwatch();
            var sw03 = new Stopwatch();
            var sw04 = new Stopwatch();
            var sw05 = new Stopwatch();

            tb_state.Text = $"draw: _x0={_x0} _y0={_y0} _res={_res}";
            
            sw01.Start();
            Cuda.Calculator.Mandelbrot(ref arr, _w * _multisampling, _h * _multisampling, _x0, _y0, _res / _multisampling, 10000, 2.0);
            //Cuda.Calculator.Buddhabrot(ref arr, _w * _multisampling, _h * _multisampling, _x0, _y0, _res / _multisampling, 10000, 10.0, 50000);
            //Cuda.Calculator.Julia(ref arr, _w * _multisampling, _h * _multisampling, _x0, _y0, _res / _multisampling, 2000, 2.0, 0.285, 0.01);
            sw01.Stop();

            if (_multisampling > 1)
                Cuda.Calculator.Multisampling(ref samples, arr, _w, _h, _multisampling);

            if (false)
            {
                Cuda.Calculator.ColorizeRGB(ref colorsRGB, _multisampling > 1 ? samples : arr, _w * _h, 0, 200);
                image.Source = BitmapFromDataRGB((int)_w, (int)_h, colorsRGB);
            }
            else
            {
                sw02.Start();
                Cuda.Calculator.ColorizeBW(ref colorsBW, _multisampling > 1 ? samples : arr, _w * _h, 0, 255);
                sw02.Stop();

                sw03.Start();
                _bitmap.Lock();
                Marshal.Copy(colorsBW, 0, _bitmapData, _w * _h);
                _bitmap.AddDirtyRect(new Int32Rect(0, 0, _w, _h));
                _bitmap.Unlock();
                sw03.Stop();
                //_bitmap.Freeze(); // for multithread

                tb_time.Text = $"time: calc={sw01.ElapsedMilliseconds}ms color={sw02.ElapsedMilliseconds}ms render={sw03.ElapsedMilliseconds}ms ";
            }
        }

        private void image_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!_mouseIn)
                return;
            _downP = e.GetPosition(image);
        }

        private void image_MouseEnter(object sender, MouseEventArgs e)
        {
            _mouseIn = true;
        }

        private void image_MouseLeave(object sender, MouseEventArgs e)
        {
            _mouseIn = false;
            tb_pos.Text = $"x={0} y={0}";
        }

        private void image_MouseMove(object sender, MouseEventArgs e)
        {
            var pos = e.GetPosition(image);
            tb_pos.Text = $"x={_x0 + pos.X * _res} y={_y0 + pos.Y * _res}";
        }

        private void image_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (!_mouseIn)
                return;
            var pos = e.GetPosition(image);
            _x0 += Math.Min(_downP.X, pos.X) * _res;
            _y0 += Math.Min(_downP.Y, pos.Y) * _res;
            _res = _res * Math.Max(Math.Abs(pos.X - _downP.X), Math.Abs(pos.Y - _downP.Y)) / _w;
            Draw();
        }

        private void image_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            var pos = e.GetPosition(image);
            double oldRes = _res;
            if (e.Delta > 0)
                _res *= 0.75;
            else
                _res /= 0.75;
            _x0 += pos.X * (oldRes - _res);
            _y0 += pos.Y * (oldRes - _res);
            Draw();
        }
    }
}