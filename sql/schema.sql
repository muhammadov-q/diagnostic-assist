-- Aurora PostgreSQL + pgvector: at ~0.4 GB of vectors, one database holds rows and vectors.

CREATE EXTENSION IF NOT EXISTS vector;

-- The cause list, per equipment family.
CREATE TABLE cause (
    tenant_id       text NOT NULL,
    cause_id        text NOT NULL,                       -- immutable once approved
    equipment_family text NOT NULL,
    label           text NOT NULL,
    kind            text NOT NULL DEFAULT 'cause',       -- cause | no_fault_found
    status          text NOT NULL,                       -- approved | proposed | retired
    merged_into     text,                                -- reviewer merges keep history
    check_minutes   integer,                             -- expert-entered: time to verify on site
    taxonomy_version text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, cause_id),
    FOREIGN KEY (tenant_id, merged_into) REFERENCES cause (tenant_id, cause_id)
);

-- One row per closed case: the output of the label factory.
CREATE TABLE case_record (
    tenant_id             text        NOT NULL,
    case_id               text        NOT NULL,
    equipment_family      text        NOT NULL,
    equipment_type        text        NOT NULL,
    created_at            timestamptz NOT NULL,
    language              text,
    symptom_text          text        NOT NULL,          -- customer_description, verbatim
    symptom_summary_en    text,                          -- LLM: customer_symptoms joined
    symptom_vec           vector(1024),                  -- embedding of symptom_text (online retrieval)
    error_codes           text[]      NOT NULL DEFAULT '{}',
    outcome_type          text,                          -- fixed | no_fault_found | replaced_without_diagnosis | not_fixed | not_technical
    label_status          text        NOT NULL,          -- trusted | provisional | no_fault_found | excluded | review
    label_reason          text,
    cause_id              text,
    component             text,
    location              text,
    mechanism             text,
    certainty             text,                          -- observed | inferred
    evidence_quote        text,
    parts                 jsonb       NOT NULL DEFAULT '[]',  -- [{part_number, role}]
    customer_report_contradicted boolean,
    issues                text[]      NOT NULL DEFAULT '{}',
    extractor_version     text,                          -- prompt + model, for selective re-extraction
    embedder_version      text,
    taxonomy_version      text,
    updated_at            timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, case_id),
    FOREIGN KEY (tenant_id, cause_id) REFERENCES cause (tenant_id, cause_id)
);

-- HNSW keeps latency flat as tenants grow; iterative_scan (pgvector >= 0.8) keeps
-- filtered queries returning k rows.
CREATE INDEX case_record_symptom_hnsw ON case_record USING hnsw (symptom_vec vector_cosine_ops);
CREATE INDEX case_record_type ON case_record (tenant_id, equipment_type) WHERE label_status IN ('trusted', 'no_fault_found');
CREATE INDEX case_record_family ON case_record (tenant_id, equipment_family) WHERE label_status IN ('trusted', 'no_fault_found');

-- Distinct (component, parts) wordings per cause: what new cases are matched against.
CREATE TABLE cause_exemplar (
    tenant_id       text NOT NULL,
    cause_id        text NOT NULL,
    component       text NOT NULL,
    part_families   text[] NOT NULL DEFAULT '{}',
    component_vec   vector(1024),
    n_cases         integer NOT NULL,
    PRIMARY KEY (tenant_id, cause_id, component, part_families),
    FOREIGN KEY (tenant_id, cause_id) REFERENCES cause (tenant_id, cause_id)
);

-- Counts the online scorer holds in memory; refreshed nightly.
CREATE MATERIALIZED VIEW cause_count AS
SELECT tenant_id, equipment_family, equipment_type, cause_id, count(*) AS n
FROM case_record
WHERE label_status IN ('trusted', 'no_fault_found')
GROUP BY 1, 2, 3, 4;

CREATE MATERIALIZED VIEW cause_part_count AS
SELECT c.tenant_id, c.equipment_type, c.cause_id, p ->> 'part_number' AS part_number, count(DISTINCT c.case_id) AS n
FROM case_record c, jsonb_array_elements(c.parts) p
WHERE c.label_status = 'trusted' AND p ->> 'role' = 'fixes_cause'
GROUP BY 1, 2, 3, 4;

-- One session per open case; the confirmed cause is the evaluation label.
CREATE TABLE diagnostic_session (
    session_id      uuid PRIMARY KEY,
    case_id         text NOT NULL,
    tenant_id       text NOT NULL,
    model_version   text NOT NULL,           -- taxonomy + scorer + embedder versions
    shown           jsonb NOT NULL,          -- every ranked list shown, with timestamps: the calibration log
    answers         jsonb NOT NULL DEFAULT '[]',
    loadout         jsonb,                   -- parts the dispatcher committed
    confirmed_cause text,                    -- technician's one tap at close
    created_at      timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, confirmed_cause) REFERENCES cause (tenant_id, cause_id)
);
