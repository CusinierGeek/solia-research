-- Solia — models.pruned_at: checkpoint retention (Sprint 7 T6, issue #134).
-- The retention policy (ai-service solia_ai.retention) deletes old checkpoint files from MODEL_DIR but
-- keeps their models row (version, metrics, history): pruned_at records when its file was deleted, so
-- the row stays traceable and a rollback to it is refused cleanly instead of failing on a missing file.
-- NULL = the checkpoint was never pruned (it may still be missing for other reasons).

ALTER TABLE models ADD COLUMN pruned_at TIMESTAMPTZ;

-- The served model's checkpoint is never pruned, and a pruned model can't be (re)crowned.
ALTER TABLE models
    ADD CONSTRAINT models_champion_not_pruned_check
        CHECK (NOT (is_champion AND pruned_at IS NOT NULL));
