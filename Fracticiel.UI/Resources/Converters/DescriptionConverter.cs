using System;
using System.ComponentModel;
using System.Globalization;
using System.Windows.Data;

namespace Fracticiel.UI.Resources.Converters;

[ValueConversion(typeof(object), typeof(string))]
public class DescriptionConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is null)
            return null;

        var type = value.GetType();

        DescriptionAttribute[] attributes = (DescriptionAttribute[])type.GetCustomAttributes(typeof(DescriptionAttribute), false);

        return attributes.Length > 0 ? attributes[0].Description : value;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}