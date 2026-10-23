# Secrets management (AES-encrypted file)

## Goal

Keep sensitive values (DB password, RPC keys, API tokens) out of source control and out of plaintext config, while remaining simple to run locally and in Docker.

## Design

- A single encrypted file `secrets/secrets.enc` holds a JSON map of secrets.
- Encryption: **AES-256-GCM** (authenticated). The master key is **never** stored in the repo.
- The master key is supplied at runtime via env var `SOLIA_MASTER_KEY` (base64, 32 bytes) — injected by the host, Docker secret, or a secret manager in production.
- On startup the backend decrypts `secrets.enc` into memory only; decrypted values never touch disk or logs.

```
secrets.enc  ──(AES-256-GCM, key=SOLIA_MASTER_KEY)──▶  { "db_password": "...", "rpc_key": "..." }
```

## File format

```
[ 12-byte nonce | ciphertext | 16-byte GCM tag ]   (base64-encoded on disk)
```

## Ownership

- The **backend (.NET)** is the reference implementation (`ISecretsProvider`, AES-GCM via `System.Security.Cryptography`). Built in Sprint 4.
- A small CLI/util to **encrypt/rotate** the file ships with the backend (`solia-secrets encrypt|rotate|get`).
- Python services read only non-sensitive config from env; any secret they need (e.g. DB password) is passed via env at container start, sourced from the same master mechanism.

## Rules

1. **Never** commit `.env`, `secrets.enc` is committed **encrypted only**; `*.key` and master keys are gitignored.
2. **Never** log decrypted secrets. The backend scrubs every `ILogger` entry and the bot's `logs` rows; the password inside `db_connection` is masked on its own too.
3. Rotate by re-encrypting with a new key; keep the previous key only long enough to re-encrypt.
4. In CI, use repository/organization secrets — never the committed file's real values.

## Well-known secret names

| Name | Effect in the backend |
|------|-----------------------|
| `db_connection` | Full Postgres connection string; replaces `SOLIA_DB_CONNECTION` / `POSTGRES_*`. |
| `db_password` | Replaces the password of the configured connection string. |

Any other name is readable via `ISecretsProvider.Get(name)`.

## Startup behaviour (backend)

| Situation | Result |
|-----------|--------|
| No `secrets.enc` at the default path (`secrets/secrets.enc`) | Plaintext config only. |
| `SOLIA_SECRETS_FILE` set but the file does not exist (typo, missing mount) | Startup fails. |
| File present, `SOLIA_MASTER_KEY` missing or not 32 base64 bytes | Startup fails. |
| Wrong key or tampered file (GCM tag mismatch) | Startup fails; nothing is decrypted. |

## Local dev quickstart

```bash
cd services/backend
export SOLIA_MASTER_KEY=$(dotnet run --project Solia.Secrets.Cli -- keygen)   # or: openssl rand -base64 32
dotnet run --project Solia.Secrets.Cli -- encrypt --in secrets.plain.json --out ../../secrets/secrets.enc
dotnet run --project Solia.Secrets.Cli -- get --name db_password --file ../../secrets/secrets.enc
dotnet run --project Solia.Secrets.Cli -- rotate --old-key "$SOLIA_MASTER_KEY" --new-key "<new b64>" --file ../../secrets/secrets.enc
# secrets.plain.json is gitignored; delete it after encrypting
```
