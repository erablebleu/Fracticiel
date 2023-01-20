namespace Fracticiel.Common.Injection;

public class Loader : ILoader
{
    private readonly List<IDisposable> _disposables = new();
    private readonly Dictionary<Type, ServiceInfo> _services = new();

    public Loader()
    {
        AddSingleton<ILoader, Loader>(this);
    }

    public void Add<T>(RegistrationMode registrationMode, T? instance = null) where T : class
    {
        Type type = typeof(T);

        _services[type] = new ServiceInfo(type, registrationMode, type.GetInterfaces().Contains(typeof(IDisposable)), instance);
    }

    public void Add<T1, T2>(RegistrationMode registrationMode, T2? instance = null) where T2 : class, T1
    {
        Type type = typeof(T1);
        Type instanceType = typeof(T2);

        _services[type] = new ServiceInfo(instanceType, registrationMode, instanceType.GetInterfaces().Contains(typeof(IDisposable)), instance);
    }

    public void AddScoped<T>() where T : class => Add<T>(RegistrationMode.Scoped);

    public void AddScoped<Tinterface, T>() where T : class, Tinterface => Add<Tinterface, T>(RegistrationMode.Scoped);

    public void AddSingleton<T>(T? instance = null) where T : class => Add<T>(RegistrationMode.Singleton, instance);

    public void AddSingleton<Tinterface, T>(T? instance = null) where T : class, Tinterface => Add<Tinterface, T>(RegistrationMode.Singleton, instance);

    public void AddTransient<T>() where T : class => Add<T>(RegistrationMode.Transient);

    public void AddTransient<Tinterface, T>() where T : class, Tinterface => Add<Tinterface, T>(RegistrationMode.Transient);

    public void Build()
    {
        foreach (var info in _services.Values)
        {
            if (info.ImplementationType.IsInterface)
                continue;
            info.InjectableProperties = info.ImplementationType.GetProperties().Where(p => Attribute.IsDefined(p, typeof(InjectableAttribute)) && _services.ContainsKey(p.PropertyType) && (p.SetMethod?.IsPublic ?? false)).ToArray();
            info.ConstructorArgs = info.ImplementationType.GetConstructors().First(c => c.IsPublic && c.GetParameters().All(p => _services.ContainsKey(p.ParameterType))).GetParameters().Select(p => p.ParameterType).ToArray();
        }
    }

    public IEnumerable<Exception> Clean()
    {
        foreach (IDisposable disposable in _disposables)
        {
            Exception? exception = null;

            try
            {
                disposable.Dispose();
            }
            catch (Exception e)
            {
                exception = e;
            }

            if (exception is null) continue;
            yield return exception;
        }
        yield break;
    }

    public T Resolve<T>() => (T)Resolve(typeof(T));

    public object Resolve(Type type)
    {
        if (!_services.ContainsKey(type))
            throw new Exception("Unregistered type");

        var info = _services[type];

        return _services[type].RegistrationMode switch
        {
            RegistrationMode.Singleton => info.Singleton ??= CreateInstance(info, true),
            RegistrationMode.Scoped or RegistrationMode.Transient => CreateInstance(info),
            _ => throw new Exception("Unknown registration moder"),
        };
    }

    private object CreateInstance(ServiceInfo info, bool setSingleton = false)
    {
        object? instance = Activator.CreateInstance(info.ImplementationType, info.ConstructorArgs.Select(t => Resolve(t)).ToArray());

        if (setSingleton)
            info.Singleton = instance;

        foreach (var prop in info.InjectableProperties)
            prop.SetValue(instance, Resolve(prop.PropertyType));

        if (instance is not null && info.IsDisposable)
            _disposables.Add((IDisposable)instance);

        return instance ?? throw new Exception($"Unable to instanciate {info.ImplementationType.FullName}.");
    }
}