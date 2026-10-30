-- Solia — bot_sessions: segment the paper trades by bot regime (Sprint 8, issue #169).
-- A session is a run of the paper bot under one configuration and one champion model. The bot opens a new
-- one when it starts (or the champion changes under it) with a fingerprint different from the current
-- session's; otherwise the current session continues across restarts. Every trade the bot opens records its
-- session, so win rate, P&L and the equity curve are read per regime instead of mixing them.
-- Written only by the bot's PostgresBotSessionStore (raw SQL); the API reads it.

CREATE TABLE bot_sessions (
    id                     BIGSERIAL   PRIMARY KEY,
    started_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Set when the next session supersedes it; NULL for the current session.
    ended_at               TIMESTAMPTZ,
    -- SHA-256 (hex) of the normalized effective options + champion version. NULL for a backfilled
    -- session, whose configuration was never recorded: the bot adopts it on its first start if the
    -- champion still matches.
    config_hash            TEXT,
    -- The effective bot options (entry rule, exits, fill model, sizing, horizon) at the session's start.
    options                JSONB,
    -- The champion when the session started; SET NULL so a model row can be deleted.
    model_id               BIGINT      REFERENCES models (id) ON DELETE SET NULL,
    model_version          TEXT,
    label                  TEXT,
    -- Open trades with an id at or below this were left by an earlier bot run (a restart empties the
    -- bot's book, so nothing will ever close them): orphans. Advanced on every bot start.
    resumed_after_trade_id BIGINT      NOT NULL DEFAULT 0,
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

-- At most one current session.
CREATE UNIQUE INDEX bot_sessions_current_idx ON bot_sessions ((TRUE)) WHERE ended_at IS NULL;

-- Nullable: the bot always sets it, but rows written outside it (fixtures, manual inserts) may not.
ALTER TABLE trades ADD COLUMN session_id BIGINT REFERENCES bot_sessions (id);
CREATE INDEX trades_session_closed_idx ON trades (session_id, closed_at);

-- Backfill: the trades opened before 2026-10-30 20:26:42 UTC used naive fills (no liquidity gate, no
-- slippage, no fees — #165), every later one realistic fills. Each session's champion is the model most of
-- its trades' predictions came from. Nothing is inserted on a database without trades.
DO $$
DECLARE
    cutover CONSTANT TIMESTAMPTZ := '2026-10-30 20:26:42+00';
    naive     BIGINT;
    realistic BIGINT;
BEGIN
    IF EXISTS (SELECT 1 FROM trades WHERE opened_at < cutover) THEN
        INSERT INTO bot_sessions (started_at, ended_at, model_id, model_version, label)
        SELECT (SELECT min(opened_at) FROM trades WHERE opened_at < cutover), cutover, m.id, m.version,
               'naive fills (pre-#165)'
        FROM (SELECT 1) AS one
        LEFT JOIN LATERAL (
            SELECT p.model_id FROM trades t JOIN predictions p ON p.id = t.prediction_id
            WHERE t.opened_at < cutover GROUP BY p.model_id ORDER BY count(*) DESC, p.model_id LIMIT 1
        ) AS top ON TRUE
        LEFT JOIN models m ON m.id = top.model_id
        RETURNING id INTO naive;

        UPDATE trades SET session_id = naive WHERE opened_at < cutover;
    END IF;

    IF EXISTS (SELECT 1 FROM trades WHERE opened_at >= cutover) THEN
        INSERT INTO bot_sessions (started_at, model_id, model_version, label)
        SELECT cutover, m.id, m.version, 'realistic fills (#165)'
        FROM (SELECT 1) AS one
        LEFT JOIN LATERAL (
            SELECT p.model_id FROM trades t JOIN predictions p ON p.id = t.prediction_id
            WHERE t.opened_at >= cutover GROUP BY p.model_id ORDER BY count(*) DESC, p.model_id LIMIT 1
        ) AS top ON TRUE
        LEFT JOIN models m ON m.id = top.model_id
        RETURNING id INTO realistic;

        UPDATE trades SET session_id = realistic WHERE opened_at >= cutover;
    END IF;
    -- With no trade after the cutover there is no current session: the bot's next start opens one.
END
$$;
