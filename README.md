# What Gets Left Behind — Reproducibility

Code, controlled pool, and reference outputs for the audit of instruction-data selectors, the
downstream analysis, and the rarity-aware correction described in the paper.

The work has three stages, each of which maps to a group of scripts here:

1. **Audit** (§3–4): what each selector keeps, measured as per-group representation ratio. No
   model training — every quantity is a property of the selected subsets.
2. **Downstream** (§5): fine-tune on selected subsets and measure whether the distortion reaches
   the trained model, and whether standard benchmarks can see it.
3. **Correction** (§6): a selector-agnostic wrapper that repartitions the budget through
   quality-gated retention floors, and a map of when each floor is necessary.

All model-based scoring uses fixed public models held constant across conditions. Fine-tuning uses
LoRA for a fixed number of epochs so any downstream difference reflects the selected data, not the
training budget.

---

## Repository layout

```
audit/                     Python package (import as `audit.*`)
  common.py                model defaults, chat templates, run metadata
  metadata/                §3.1–3.2  controlled pool construction + group labels
    build_metadata.py        assemble metadata (resource tier + skill) for a pool
    inject_muri.py           enrich the base sample with native low-resource examples
    langid.py                GlotLID language identification
    pool_io.py               canonical pool read/write
    export_pilot_jsonl.py    export the labeled controlled pool
  selectors/               §3.3  the eight selectors (one representative per signal family)
    run_random.py            group-neutral control / fair-share reference
    run_perplexity.py        keep-high / keep-low / keep-middle (one scorer, three directions)
    run_ifd.py               instruction-following difficulty
    run_semdedup.py          semantic deduplication (reuses the RDS+ pool embeddings)
    run_quality.py           LLM-judged quality (records per-example 1–5 ratings)
    run_rdsplus.py           representation-similarity retrieval
  metrics/                 §3.4, §4  the representation-ratio audit
    audit_metrics.py         representation ratio, retention, coverage
    run_audit.py             audit a directory of selections -> table + faceted figures
    quality_score_by_tier.py mean judge rating by resource tier (Table 2)
  stagec/                  §6  the correction
    rarity_aware.py          the wrapper: quality gate + retention floors + within-group ranking
    materialize_matrix.py    materialize floored subsets across selectors/floors/budgets
    materialize_necessity.py necessity-map cells (language + skill axes)
    materialize_partb.py     code + safety cells (Part B)
    materialize_crossmodel.py model-independent cells for the cross-model replication
    materialize_stagec_variants.py shared subset helpers
    build_stagec_pool.py     build the density-enriched correction pool
  stageb/                  §5  downstream training + evaluation
    train_one_cell.py        LoRA fine-tune one selected subset
    heldout_ppl.py           held-out FLORES perplexity on native text
    run_eval.py              lm-eval harness wrapper (Belebele / chrF++ / MMLU / GSM8K / IFEval / MBPP)
    make_flores_tasks.py     FLORES chrF++ task definitions
    define_eval_languages.py the above-chance gate + evaluated-language config
    build_concentrated_pool.py build the language-density pool for the erosion tables
    materialize_round4.py    materialize the floor-free downstream subsets
  experiments/            top-level drivers
    run_stage_a.py           run all selectors -> selection JSONs (feeds the audit)
    run_round4_matrix.py     train the floor-free downstream subsets (language axis)
    run_round4_eval.py       evaluate them (perplexity + chrF++ + Belebele + MMLU)
    run_stagec_train.py      train the correction matrix on the base model
    run_stagec_eval.py       held-out perplexity evaluation of the correction cells
    run_skill_cells_eval.py  skill-benchmark evaluation of trained cells
    run_movability_check.py  base-vs-full movability gate for skill benchmarks
    eval_xstest.py           XSTest safety evaluation (both sub-scores)
    assemble_matrix.py       assemble the correction results table
    assemble_crossmodel.py   assemble the cross-model replication table
    compute_accounting.py    per-group retention accounting
  configs/
    joshi_resource.json      six-tier resource taxonomy
    dataset_to_skill.json    provenance -> skill map
    quality_judge_prompt.txt the language-neutral judge rubric
    n_abs_by_axis.json       absolute-floor size per group (language / skill)
    eval_languages.json      evaluated languages; eval_languages_decisive.json is the decisive set
    round4_decisive.json     decisive-language injection spec
    belebele_languages.json  reading-comprehension language map
    build_metadata.yaml      metadata build settings
    stageb_train.yaml        LoRA / training hyperparameters
    flores_tasks/            generated FLORES task files

data/
  controlled_pool.jsonl          the controlled pool, N=10,842 (Table 1)
  controlled_pool_metadata.parquet per-example group labels (tier + skill), fixed before selection
  selections/                    cached selection outputs (selector x budget) for the audit

results/                         reference numbers behind each table/figure
  audit_representation_ratio_N10842.csv   full RR table (all selectors x axes x budgets)
  fig1_rr_skill_b0.05.csv                 Figure 2 data (skill axis, 5%)
  fig2_rr_resource_b0.05.csv              Figure 1 data (resource tier, 5%)
  table2_quality_judge_by_tier.csv        Table 2 (mean judge rating by tier)
  table3_language_downstream.csv          Table 3 (language-axis downstream)
  table5_metric_blindness.csv             Table 5 (three measurements, same comparison)
  above_chance_gate.csv                   per-language base accuracy + gate verdict
  code_recovery.csv, safety_calibration.csv  Part B (code HumanEval, safety XSTest)
  crossmodel_results.parquet              cross-model replication (§6.6)

requirements.txt
```

---

## Reproducing the results

Install dependencies (Python 3.11; a CUDA GPU is required for §5–6, not for §3–4):

```bash
pip install -r requirements.txt
```

### Stage A — the audit (Tables 1–2, Figures 1–2)

The audit needs no training. Given a labeled pool and a directory of selection JSONs, `run_audit`
computes every representation ratio and renders the faceted figures:

```bash
python -m audit.metrics.run_audit \
    --metadata data/controlled_pool_metadata.parquet \
    --selections_dir data/selections \
    --output_dir results/audit
```

To regenerate the selection JSONs from scratch, run every selector on the controlled pool. Each
selector writes `<selector>__b<budget>__s<seed>.json`; scoring uses the fixed public scorers named
in `common.py` (a small multilingual LM for perplexity/IFD, a strong instruct model for the quality
judge, a 7B backbone for RDS+):

```bash
python -m audit.experiments.run_stage_a \
    --pool data/controlled_pool.jsonl \
    --metadata data/controlled_pool_metadata.parquet \
    --selections_dir data/selections
```

The mean judge rating by resource tier (Table 2) is recovered from the judge's saved per-example
ratings:

```bash
python -m audit.metrics.quality_score_by_tier --metadata data/controlled_pool_metadata.parquet
```

### Stage B — downstream (Tables 3–5)

Build the density-enriched language pool, materialize the floor-free subsets, train, and evaluate.
`train_one_cell` fine-tunes one subset with LoRA; `run_round4_eval` reports held-out FLORES
perplexity, chrF++, and Belebele as decisive-language macros. The base model is a capable
multilingual model that clears the above-chance gate (`define_eval_languages.py`).

```bash
python -m audit.stageb.build_concentrated_pool
python -m audit.experiments.run_round4_matrix --gpus 0,1,2
python -m audit.experiments.run_round4_eval  --gpus 0,1,2
```

The skill-axis downstream (Table 4) uses the skill-density pool (`build_stagec_pool.py`) with
`run_skill_cells_eval` for GSM8K / IFEval / MBPP / HumanEval.

### Stage C — the correction (Tables 6–7)

The wrapper is `stagec/rarity_aware.py`: a group-agnostic quality gate, a retention floor
(`none` / `proportional` / `absolute` / `hybrid`, sizes from `configs/n_abs_by_axis.json`), then
within-group selection by the base selector's own ranking. Materialize the floored cells, train,
evaluate, assemble:

```bash
python -m audit.stagec.materialize_matrix --stage_a <stage_a_scores_dir>
python -m audit.experiments.run_stagec_train --gpus 0,1,2 --target_model <base>
python -m audit.experiments.run_stagec_eval  --gpus 0,1,2 --ppl_only
python -m audit.experiments.assemble_matrix
```

Part B (code + safety) uses `materialize_partb.py` + `run_skill_cells_eval` (HumanEval) and
`eval_xstest.py` (both XSTest sub-scores). The cross-model replication (§6.6) uses
`materialize_crossmodel.py` (the cells are model-independent) trained on a second base, assembled
with `assemble_crossmodel.py`.

---

## Notes on cached artifacts

To keep the release small, three kinds of large intermediate are not shipped and are regenerated by
the commands above:

- **Selector score caches** (perplexity NLLs, IFD scores, quality ratings, RDS+ embeddings). These
  are produced once per pool by the selector scripts and cached; the audit and the wrapper both read
  them. `data/selections/` contains the resulting selection JSONs for the audit.
- **Trained LoRA adapters** for each downstream cell.
- **Evaluation working directories** from the harness.

`results/` holds the exact numbers behind every table and figure so the reported values can be
checked without rerunning training.

## Environment

All training and evaluation were run on a single workstation with three 24 GB GPUs; each
fine-tuning run uses one GPU. Selector scoring, held-out perplexity, and all benchmark evaluation
use fixed public models and a standard open evaluation harness, with generation settings held
constant across conditions. Exact model identifiers, budgets, seeds, and hyperparameters are in
`common.py`, `configs/`, and the training YAML.
