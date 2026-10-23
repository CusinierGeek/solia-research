using Solia.Secrets.Cli;

return SecretsCli.Run(args, Console.Out, Console.Error, Environment.GetEnvironmentVariable);
