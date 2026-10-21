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
2. **Never** log decrypted secrets (add a redaction filter).
3. Rotate by re-encrypting with a new key; keep the previous key only long enough to re-encrypt.
4. In CI, use repository/organization secrets — never the committed file's real values.

## Local dev quickstart (once Sprint 4 lands)

```bash
export SOLIA_MASTER_KEY=$(openssl rand -base64 32)
solia-secrets encrypt --in secrets.plain.json --out secrets/secrets.enc
# secrets.plain.json is gitignored; delete after encrypting
```
