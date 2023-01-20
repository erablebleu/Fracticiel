namespace Fracticiel.Common.Injection;

public interface ILoader
{
    T Resolve<T>();
}