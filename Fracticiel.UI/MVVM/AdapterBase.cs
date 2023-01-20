using System.Runtime.CompilerServices;

namespace Fracticiel.UI.MVVM;

public abstract class AdapterBase : ObservableObject
{
    private bool _isDirty;
    public bool IsDirty { get => _isDirty; set => Set(ref _isDirty, value); }

    public bool Set<T>(ref T obj, T value, [CallerMemberName] string propertyName = null, bool dirty = false)
    {
        if (obj != null && obj.Equals(value))
            return false;

        obj = value;
        RaisePropertyChanged(propertyName);

        if (dirty)
            IsDirty = true;

        return true;
    }

    public bool SetDirty<T>(ref T obj, T value, [CallerMemberName] string propertyName = null)
        => Set(ref obj, value, propertyName, true);
}