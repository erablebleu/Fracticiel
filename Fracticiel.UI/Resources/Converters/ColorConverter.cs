using System;
using System.Globalization;
using System.Windows.Data;

namespace Fracticiel.UI.Resources.Converters;

[ValueConversion(typeof(System.Drawing.Color), typeof(System.Windows.Media.Color))]
public class ColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not System.Drawing.Color color)
            return System.Windows.Media.Colors.Black;

        return System.Windows.Media.Color.FromArgb(color.A, color.R, color.G, color.B);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not System.Windows.Media.Color color)
            return System.Drawing.Color.Black;

        return System.Drawing.Color.FromArgb(color.A, color.R, color.G, color.B);
    }
}