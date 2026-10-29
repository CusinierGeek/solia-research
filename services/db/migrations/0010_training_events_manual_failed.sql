-- Solia — training_events: manual trigger + failed decision (Sprint 7 T2, issue #130).
-- The retrain orchestrator records every run, including the ones the backend kicks off through
-- POST /model/retrain ({"trigger": "manual"}) and the ones that crash or error. A failed run gets
-- finished_at + decision 'failed': it trained no usable model, so it must not be mistaken for a real
-- outcome nor reset the trades-since-last-train count (only promoted / rejected do; a rollback only
-- restores the previous champion, so it does not either).
-- The column CHECKs of 0001 carry PostgreSQL's default names; they are replaced in place.

ALTER TABLE training_events
    DROP CONSTRAINT training_events_trigger_check,
    ADD CONSTRAINT training_events_trigger_check
        CHECK (trigger IN ('initial', 'scheduled', 'n_trades', 'manual')),
    DROP CONSTRAINT training_events_decision_check,
    ADD CONSTRAINT training_events_decision_check
        CHECK (decision IN ('promoted', 'rejected', 'rollback', 'failed'));

-- A run's decision and finished_at are written together: both NULL while it is in flight, both set
-- once it ends. Rows written before this rule are brought in line first: a finished run without a
-- decision has no known outcome, so it is recorded as failed (it must not pass for a trained model);
-- a decided run without finished_at is closed at its own start.
UPDATE training_events SET decision = 'failed' WHERE finished_at IS NOT NULL AND decision IS NULL;
UPDATE training_events SET finished_at = started_at WHERE finished_at IS NULL AND decision IS NOT NULL;

ALTER TABLE training_events
    ADD CONSTRAINT training_events_finished_decision_check
        CHECK ((decision IS NULL) = (finished_at IS NULL));
