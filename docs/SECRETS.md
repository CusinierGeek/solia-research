# Secrets management (AES-encrypted file)

## Goal

Keep the backend's sensitive values (RPC keys, API tokens, its DB credentials) out of source control and out of plaintext config, while remaining simple to run locally and in Docker.

> **Scope:** only the backend reads `secrets.enc`. Postgres itself, `db-migrate`, the data-pipeline and
> the ai-service take the DB password from `POSTGRES_PASSWORD` (env), so storing `db_password` in
> `secrets.enc` does **not** remove it from the environment — it stays in plaintext `.env` (or, better,
> the host env / a secret manager in production). Don't delete `POSTGRES_PASSWORD` from `.env` after
> moving it into `secrets.enc`: every container would silently fall back to the dev default `change_me`.

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
- Python services read only non-sensitive config from env; any secret they need (e.g. the DB password) is passed via env at container start (`POSTGRES_PASSWORD`) — they don't read `secrets.enc`.

## Rules

1. **Never** commit `.env`, `config/*.env`, any `secrets/` folder (including the encrypted `secrets.enc` — it is mounted at runtime, not versioned), `*.key` or master keys: all gitignored, and dockerignored so they never reach an image. (Only `secrets/` is ignored, not every `*.enc`, so encrypted test fixtures stay committable.)
2. **Never** log decrypted secrets. The backend scrubs every `ILogger` entry and the bot's `logs` rows; the password inside `db_connection` is masked on its own too.
3. Rotate by re-encrypting with a new key; keep the previous key only long enough to re-encrypt.
4. In CI, use repository/organization secrets or a throwaway key generated in the job — never real values.

## Well-known secret names

| Name | Effect in the backend |
|------|-----------------------|
| `db_connection` | Full Postgres connection string; replaces `SOLIA_DB_CONNECTION` / `POSTGRES_*`. |
| `db_password` | Replaces the password of the configured connection string (backend only — see *Scope* above). |

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
# Build once, then run with --no-build: `dotnet run` may print restore/build output on stdout,
# which `$(…)` would capture into the key.
dotnet build -v q Solia.Secrets.Cli
cli() { dotnet run --no-build --project Solia.Secrets.Cli -- "$@"; }
export SOLIA_MASTER_KEY=$(cli keygen)   # or: openssl rand -base64 32
cli encrypt --in secrets.plain.json --out ../../secrets/secrets.enc
cli get --name rpc_key --file ../../secrets/secrets.enc
cli rotate --old-key "$SOLIA_MASTER_KEY" --new-key "<new b64>" --file ../../secrets/secrets.enc
# secrets.plain.json is gitignored; delete it after encrypting
```

## Docker (compose)

The backend is the **only** container that receives `SOLIA_MASTER_KEY` or the secrets file; every
other service gets an explicit per-service variable list (compose interpolation; the only `env_file`
is each service's own optional `config/<service>.env` — don't put the master key in another service's).

```bash
# 1. Create the key and the encrypted file on the host (the CLI is not in any image).
dotnet build -v q services/backend/Solia.Secrets.Cli      # then --no-build: keeps build output out of $(…)
cli() { dotnet run --no-build --project services/backend/Solia.Secrets.Cli -- "$@"; }
export SOLIA_MASTER_KEY=$(cli keygen)
mkdir -p secrets
cli encrypt --in secrets.plain.json --out secrets/secrets.enc
rm secrets.plain.json
chmod 644 secrets/secrets.enc        # readable by the backend's non-root UID 1654 (or chown 1654 + 600)

# 2. Inject the key — host env (preferred; wins over .env) or SOLIA_MASTER_KEY=… in .env.
docker compose up -d --build
```

- **Mount**: `./secrets` → `/app/secrets` **read-only**; the backend reads the default path
  `/app/secrets/secrets.enc` (`SOLIA_SECRETS_FILE` is not passed to the container).
- **No file** ⇒ the stack starts on plaintext config (`.env` / `POSTGRES_*`).
- `POSTGRES_PASSWORD` stays in `.env` / the host env either way (see *Scope*): postgres and the other
  services need it.
- **File + missing/wrong key** ⇒ the backend exits non-zero at startup: `docker compose ps` shows it
  `Restarting (1)` (the dashboard, which waits for a healthy backend, doesn't start) and
  `docker compose logs backend` shows the `SecretsException` reason (e.g. *failed authentication
  (wrong master key or tampered file)*) — never the key or a decrypted value.
- Rotate: `rotate --old-key … --new-key …` on the host, update the injected key, `docker compose up -d backend`.
