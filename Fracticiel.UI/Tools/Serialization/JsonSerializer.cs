using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Fracticiel.UI.Tools.Serialization;

public static class JsonSerializer
{
    private static System.Text.Json.JsonSerializerOptions Options = new();

    static JsonSerializer()
    {
    }

    public static T? Deserialize<T>(string value) => System.Text.Json.JsonSerializer.Deserialize<T>(value, Options);

    public static T? Deserialize<T>(Stream utf8Json)
        => System.Text.Json.JsonSerializer.DeserializeAsync<T>(utf8Json, Options).AsTask().GetAwaiter().GetResult();

    public static ValueTask<T?> DeserializeAsync<T>(Stream utf8Json, CancellationToken cancellationToken = default)
            => System.Text.Json.JsonSerializer.DeserializeAsync<T>(utf8Json, Options, cancellationToken);

    public static string Serialize<T>(T value) => System.Text.Json.JsonSerializer.Serialize(value, Options);

    public static void Serialize<T>(Stream utf8Json, T value)
            => System.Text.Json.JsonSerializer.SerializeAsync(utf8Json, value, Options).GetAwaiter().GetResult();

    public static Task SerializeAsync<T>(Stream utf8Json, T value, CancellationToken cancellationToken = default)
                => System.Text.Json.JsonSerializer.SerializeAsync(utf8Json, value, Options, cancellationToken);
}