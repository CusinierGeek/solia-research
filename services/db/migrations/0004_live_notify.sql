-- Solia — live change notifications (Sprint 5, SignalR live hub).
-- Every new prediction, opened/closed trade and non-debug log row raises a NOTIFY on channel `solia_live` with the
-- payload '<event>:<id>', whichever process wrote it (the backend's bot, the data-pipeline, the ai-service).
-- The backend LISTENs, re-reads the row by id (the table stays the source of truth) and pushes it to the
-- dashboard. NOTIFY is delivered on commit only and never waits for a listener: with nobody listening the
-- notification is simply dropped, so writers are never slowed down by the live feed.
-- <event> is the hub's event name (docs/API-CONTRACTS.md, "Live hub").

CREATE FUNCTION solia_notify_live() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('solia_live', TG_ARGV[0] || ':' || NEW.id);
    RETURN NULL;
END
$$;

CREATE TRIGGER predictions_notify_created AFTER INSERT ON predictions
    FOR EACH ROW EXECUTE FUNCTION solia_notify_live('prediction_created');

-- A trade is normally inserted open and closed by an UPDATE; one inserted already closed is announced closed.
CREATE TRIGGER trades_notify_opened AFTER INSERT ON trades
    FOR EACH ROW WHEN (NEW.status = 'open') EXECUTE FUNCTION solia_notify_live('trade_opened');

CREATE TRIGGER trades_notify_inserted_closed AFTER INSERT ON trades
    FOR EACH ROW WHEN (NEW.status = 'closed') EXECUTE FUNCTION solia_notify_live('trade_closed');

CREATE TRIGGER trades_notify_closed AFTER UPDATE OF status ON trades
    FOR EACH ROW WHEN (OLD.status = 'open' AND NEW.status = 'closed') EXECUTE FUNCTION solia_notify_live('trade_closed');

-- Debug rows are not announced: they can be high-volume and each announced row costs the backend a re-read.
CREATE TRIGGER logs_notify_written AFTER INSERT ON logs
    FOR EACH ROW WHEN (NEW.level <> 'debug') EXECUTE FUNCTION solia_notify_live('log_written');
