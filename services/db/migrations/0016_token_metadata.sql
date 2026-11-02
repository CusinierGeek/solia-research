-- Solia — token metadata features (feature enrichment FE-T12, issue #196).
-- At launch a token's only information is its metadata. Two kinds of signal, both point-in-time:
--   * stream-derived (free, deterministic, live = replay): whether the create's metadata URI is content-addressed
--     (IPFS / Arweave, the pump.fun UI default) or a launch tool's own host, and the narrative heat of its
--     normalized name / ticker — other launches sharing it in the last hour, tokens sharing it that graduated in the
--     last 3 h. Only whole seconds strictly before the row's ts count; the counts stay NULL until
--     meta_coverage_sec reaches 1 h (matches) / 6 h (graduations): replay warm-up, or a live restart.
--   * fetched once per token by live ingestion (rate-limited, in the background, never during a replay): whether
--     the metadata JSON lists a website / X / Telegram link. A row only sees a fetch completed by its ts
--     (token_metadata.fetched_at <= ts); NULL when not known by then (not fetched yet, failed, or a replay of a
--     token the cache doesn't hold — any window before the fetcher was deployed).
-- Flags are SMALLINT 0/1 so every feature column casts to float8. meta_coverage_sec is always written, so a NULL
-- there marks a row written before this migration.

ALTER TABLE features
    ADD COLUMN meta_coverage_sec          INT,      -- seconds of stream history behind the narrative counts, capped at 21600
    ADD COLUMN meta_uri_content_addressed SMALLINT, -- 1 = IPFS / Arweave URI, 0 = launch tool's own host, NULL = no create / URI seen
    ADD COLUMN meta_has_website           SMALLINT, -- metadata lists a website (NULL = not known by ts)
    ADD COLUMN meta_has_twitter           SMALLINT, -- metadata lists an X / Twitter link (NULL = not known by ts)
    ADD COLUMN meta_has_telegram          SMALLINT, -- metadata lists a Telegram link (NULL = not known by ts)
    ADD COLUMN meta_name_matches_1h       INT,      -- other launches in the last hour sharing a normalized name / ticker
    ADD COLUMN meta_narrative_grads_3h    INT;      -- other tokens sharing a normalized name / ticker that graduated in the last 3 h

-- The single metadata fetch of each token: its outcome, when it completed, and the three link flags. Nothing else
-- from the metadata (URI, links, description, image) is stored. Written by live ingestion only; replays read it.
CREATE TABLE IF NOT EXISTS token_metadata (
    mint         TEXT        PRIMARY KEY,
    fetched_at   TIMESTAMPTZ NOT NULL,              -- when the fetch completed: a snapshot sees the flags only from then
    status       SMALLINT    NOT NULL,              -- 1 = ok, 2 = failed (unreachable, timeout, non-2xx, too large, malformed), 3 = unsupported URI
    has_website  BOOLEAN     NOT NULL DEFAULT FALSE,
    has_twitter  BOOLEAN     NOT NULL DEFAULT FALSE,
    has_telegram BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT token_metadata_status_check CHECK (status IN (1, 2, 3))
);

-- Startup loads the results of a fetch-time range (live: the last 48 h; replay: the replayed span).
CREATE INDEX IF NOT EXISTS token_metadata_fetched_at_idx ON token_metadata (fetched_at);
