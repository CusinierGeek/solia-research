using System.Security.Cryptography;

namespace Solia.Secrets.Cli;

/// <summary>
/// <c>solia-secrets</c>: encrypts, rotates and reads <c>secrets.enc</c> (docs/SECRETS.md). All crypto goes
/// through <see cref="SecretsFile"/>; plaintext is never written to disk. Errors name keys and paths, never
/// values.
/// </summary>
public static class SecretsCli
{
    public const int Ok = 0;
    public const int Failure = 1;
    public const int Usage = 2;

    private const string UsageText = """
        Usage:
          solia-secrets encrypt --in <plain.json> --out <secrets.enc>      (key: SOLIA_MASTER_KEY)
          solia-secrets rotate  --old-key <b64> --new-key <b64> --file <secrets.enc>
          solia-secrets get     --name <secret> [--file <secrets.enc>]      (key: SOLIA_MASTER_KEY)
          solia-secrets keygen                                              (prints a new base64 key)
        """;

    /// <summary>Each command's accepted options; anything else is a usage error, never silently ignored.</summary>
    private static readonly Dictionary<string, string[]> Commands = new(StringComparer.Ordinal)
    {
        ["encrypt"] = ["in", "out"],
        ["rotate"] = ["old-key", "new-key", "file"],
        ["get"] = ["name", "file"],
        ["keygen"] = [],
    };

    /// <param name="env">Reads an environment variable (injected for tests).</param>
    public static int Run(string[] args, TextWriter stdout, TextWriter stderr, Func<string, string?> env)
    {
        if (args.Length == 0)
        {
            stderr.WriteLine(UsageText);
            return Usage;
        }

        try
        {
            string command = args[0];
            if (!Commands.TryGetValue(command, out string[]? allowed))
            {
                throw new UsageException($"Unknown command '{command}'.");
            }

            Dictionary<string, string> options = ParseOptions(command, args.AsSpan(1), allowed);
            return command switch
            {
                "encrypt" => Encrypt(options, stdout, env),
                "rotate" => Rotate(options, stdout),
                "get" => Get(options, stdout, env),
                _ => Keygen(stdout),
            };
        }
        catch (UsageException ex)
        {
            stderr.WriteLine(ex.Message);
            stderr.WriteLine(UsageText);
            return Usage;
        }
        catch (SecretsException ex)
        {
            stderr.WriteLine($"error: {ex.Message}");
            return Failure;
        }
        catch (IOException ex)
        {
            stderr.WriteLine($"error: {ex.Message}");
            return Failure;
        }
        catch (UnauthorizedAccessException ex)
        {
            stderr.WriteLine($"error: {ex.Message}");
            return Failure;
        }
    }

    private static int Encrypt(Dictionary<string, string> options, TextWriter stdout, Func<string, string?> env)
    {
        string input = Required(options, "in");
        string output = Required(options, "out");
        if (!File.Exists(input))
        {
            throw new SecretsException($"Input file '{input}' does not exist.");
        }

        byte[] plaintext = File.ReadAllBytes(input);
        try
        {
            Dictionary<string, string> secrets = SecretsFile.ParseJson(plaintext);
            WithKey(env(MasterKey.EnvironmentKey), MasterKey.EnvironmentKey, key =>
            {
                SecretsFile.Write(output, secrets, key);
                return true;
            });
            stdout.WriteLine($"Encrypted {secrets.Count} secret(s) to {output}.");
            return Ok;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(plaintext);
        }
    }

    private static int Rotate(Dictionary<string, string> options, TextWriter stdout)
    {
        string oldKey = Required(options, "old-key");
        string newKey = Required(options, "new-key");
        string file = Required(options, "file");

        // Both keys are validated (and the file decrypted) before anything is written, so a typo or a wrong
        // old key leaves the file untouched.
        byte[] oldBytes = MasterKey.Parse(oldKey, "--old-key");
        byte[]? newBytes = null;
        try
        {
            newBytes = MasterKey.Parse(newKey, "--new-key");
            Dictionary<string, string> secrets = SecretsFile.Read(file, oldBytes);
            SecretsFile.Write(file, secrets, newBytes);
            stdout.WriteLine($"Rotated {secrets.Count} secret(s) in {file} to the new key.");
            return Ok;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(oldBytes);
            if (newBytes is not null)
            {
                CryptographicOperations.ZeroMemory(newBytes);
            }
        }
    }

    private static int Get(Dictionary<string, string> options, TextWriter stdout, Func<string, string?> env)
    {
        string name = Required(options, "name");
        string file = options.GetValueOrDefault("file")
            ?? NonBlank(env(SecretsProvider.FileKey))
            ?? SecretsFile.DefaultPath;

        SecretsProvider secrets = WithKey(
            env(MasterKey.EnvironmentKey), MasterKey.EnvironmentKey, key => new SecretsProvider(SecretsFile.Read(file, key)));
        stdout.WriteLine(secrets.Get(name));
        return Ok;
    }

    private static int Keygen(TextWriter stdout)
    {
        stdout.WriteLine(MasterKey.Generate());
        return Ok;
    }

    private static T WithKey<T>(string? base64, string source, Func<byte[], T> use)
    {
        byte[] key = MasterKey.Parse(base64, source);
        try
        {
            return use(key);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
        }
    }

    /// <summary>
    /// Parses <c>--name value</c> pairs, rejecting options <paramref name="command"/> doesn't take (a typo such
    /// as <c>--fiel</c> would otherwise fall back to the default file) and repeated ones (which would otherwise
    /// silently keep the last).
    /// </summary>
    private static Dictionary<string, string> ParseOptions(string command, ReadOnlySpan<string> args, string[] allowed)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || args[i].Length == 2)
            {
                throw new UsageException($"Unexpected argument '{args[i]}'.");
            }
            if (i + 1 >= args.Length)
            {
                throw new UsageException($"Option '{args[i]}' needs a value.");
            }
            string name = args[i][2..];
            if (!allowed.Contains(name, StringComparer.Ordinal))
            {
                throw new UsageException($"Unknown option '{args[i]}' for '{command}'.");
            }
            if (!options.TryAdd(name, args[i + 1]))
            {
                throw new UsageException($"Option '{args[i]}' is given more than once.");
            }
        }
        return options;
    }

    private static string Required(Dictionary<string, string> options, string name) =>
        NonBlank(options.GetValueOrDefault(name)) ?? throw new UsageException($"Missing required option --{name}.");

    private static string? NonBlank(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim();

    private sealed class UsageException(string message) : Exception(message);
}
