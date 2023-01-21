using AutoMapper;
using Fracticiel.Common.Injection;

namespace Fracticiel.UI.MVVM;

public abstract class ViewModelBase : AdapterBase
{
    [Injectable] public ILoader? Loader { get; set; }
    [Injectable] public IMapper Mapper { get; set; }

    public virtual void Load()
    {
    }
}