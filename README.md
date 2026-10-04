# Diagnostic Assist: the label factory

The full write-up is in **[docs/submission.pdf](docs/submission.pdf)**: the product, the design, how I evaluated it,
what I cut, next steps, and where I used AI. The LaTeX source is in the same folder.

This README is about the code. I built one part of the system: the offline pipeline that reads old closed cases,
which have no labels, and builds a list of root causes for each equipment family.

## Run it

You need Python 3.11 or newer, with `numpy` and `scipy`. You don't need an API key. By default, the scripts replay
real model output that I recorded once (`fixtures/extractions_model.json`, Claude Opus 5.5) instead of calling the LLM.

Run all commands from the project root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # the 63 tests (pytest works too)
python3 scripts/run_labeling.py                        # label the 22 sample cases, then add later cases as new ones
python3 scripts/evaluate_against_gold.py               # score the result against my hand labels
```

Both scripts take `--extractions fixtures/extractions_sample.json` to replay my hand-written extractions instead.

To record the model output again (needs the Claude Code CLI, about $0.30):

```bash
python3 scripts/record_extractions.py                  # one call per case, same prompt and schema as the extractor
```

`evaluate_against_gold.py` reports B-cubed precision and recall at different merge thresholds, with and without the
parts rule.

## Try it on new cases

```bash
python3 scripts/label_case.py examples/*.json
```

There are three example cases, one for each possible result:

| File | What happens |
|---|---|
| `examples/joins_existing_cause.json` | A French case on a CX-300. It joins the existing "main contactor" cause, which so far has one English and one German case. |
| `examples/proposes_new_cause.json` | A tank safety valve, which the sample has never seen. It is not forced into the nearest "valve" cause. It becomes a *proposed* new cause that an expert has to approve. |
| `examples/sent_to_review.json` | The extraction is wrong on purpose: it takes the cause from the customer's description, not from the technician's notes. The checks catch this and hold the case for review. |

For each case, the script prints:

- what was extracted
- each check, and whether it passed
- the status
- the result: the case joined an existing cause (and you see what that cause looks like now), became a proposed new
  cause, or was held for review

### Try your own case

Copy one of the example files, give it a new `case_id`, and change the `case` part. The file also has an `extraction`
part. This is what the LLM would normally return, so you fill it in yourself and play the model:

```bash
python3 scripts/label_case.py my_case.json
```

If your case is held for review, the "Checks" part of the output tells you why. The most common reason is that the
`evidence_quote` does not appear word for word in the technician's notes or the resolution text.

## Output

The results of the last run are in `output/`:

- `record_extractions.txt`: the model, effort and cost of the recorded run
- `run_labeling.txt` and `evaluate_against_gold.txt`: what the two scripts printed, on the real model output
- `case_records.jsonl`: one `case_record` row per case
- `cause_registry.json`: the cause list

## Layout

```
src/diagnostic_assist/
  domain.py               data records: Case, Extraction, LabeledOutcome, enums
  config.py               every number you can tune, and why it has that value
  labeling/
    triage.py             skips cases with no technician text
    prompt.py             the extraction prompt and its JSON schema
    extractors.py         the LLM call (one case at a time; never run live), and replay of saved extractions
    normalize.py          error codes, part families, transliteration, a stub parts catalogue
    validate.py           the checks: evidence quote, parts, outcome, error codes
    labeler.py            parts catalogue override, and the status decision
    embeddings.py         offline text embedder that gives more weight to the main noun; Bedrock stub
    canonicalize.py       clustering per family, the cause list, nightly assignment, the cause table
    quality.py            parts purity, same-fix-parts pairs, language-split test, status mix, B-cubed
    pipeline.py           backfill() and update()
fixtures/
  sample_cases.json       the 22 cases from the brief, unchanged
  extractions_model.json  real model output for the 22 cases (scripts/record_extractions.py)
  extractions_sample.json extractions I wrote by hand, with deliberate variations; the tests use these
  gold_causes.json        my hand labels, used by evaluate_against_gold
examples/                 new cases for scripts/label_case.py
scripts/                  record_extractions, run_labeling, evaluate_against_gold, label_case
sql/schema.sql            the Postgres + pgvector tables the pipeline writes to
tests/                    unit and end-to-end tests; every trap in the sample data has its own check
docs/                     the brief (.docx), the submission (PDF) and its LaTeX source
```

## Where to start reading

1. `labeling/canonicalize.py`, the docstring at the top: how specific a cause should be, and why I chose this way of
   clustering.
2. `labeling/validate.py`: what stops the LLM from quietly making up labels.
3. `tests/test_pipeline_sample.py`: the traps in the sample data, with a check for each one.
4. `config.py`: every threshold, and why it has its value.

## Limitations

- I had no API key, so `fixtures/extractions_model.json` was recorded through the Claude Code CLI with the same
  prompt and schema; the SDK call in `extractors.py` is written but has not run. On the real output, the 22 cases group
  exactly like my hand labels. The one disagreement is C-48801, a temporary fix: the model gives no provisional cause
  where I gave one, which exposed an ambiguity in rule 6 of the prompt (section 3 of the submission).
- `fixtures/extractions_sample.json` is hand-written and stays the test fixture: it includes small, realistic
  differences on purpose (umlauts and accents in quotes, different names for the same component) to test the evidence
  check and the merging.
- 22 clean cases show that the mechanics work, not that the results are accurate. Nothing reached review, so the
  checks have not met a real model error yet, and the threshold sweep shows this sample is too small to choose a
  threshold. See section 3 of the submission.
- Some parts are stubs: the ERP parts catalogue (`PartCatalogue`), the Bedrock embedder, the batch extraction for the
  backfill, and mapping re-clustered causes back to their existing IDs. Each stub has a docstring that explains how
  the real version would work. Saving to the database, job scheduling and the online scorer are designed (section 2
  of the submission) but not built.
