using System.Reflection;

namespace Fracticiel.Common.Injection;

internal class ServiceInfo
{
    public Type[] ConstructorArgs = Array.Empty<Type>();
    public Type ImplementationType;
    public PropertyInfo[] InjectableProperties = Array.Empty<PropertyInfo>();
    public bool IsDisposable;
    public RegistrationMode RegistrationMode;
    public object? Singleton;

    public ServiceInfo(Type implementationType, RegistrationMode registrationMode, bool isDisposable, object? singleton)
    {
        ImplementationType = implementationType;
        RegistrationMode = registrationMode;
        IsDisposable = isDisposable;
        Singleton = singleton;
    }
}