# HANDOFF: build and evaluate the HSTU on a GPU server

You are taking over a project. Data preparation, splits and four baselines are done. Your job is the HSTU model, its comparison models and ablations, evaluation, biology checks, plots, and README. Read this whole file first, then `PLAN.md` (history and change log).

**Deadline: Sunday Oct 18, 3pm PT (22:00 UTC)**, with final results, plots and a demo. Report results honestly, including if HSTU only ties GBSA.

**Ground rules:** data never leaves SPC AWS (use only the buckets below, no external APIs with patient data); do not attempt re-identification or link individuals across datasets; do not use the Sid Sijbrandij osteosarcoma dataset as a holdout (decided: not usable, single patient, ground-rule risk).

## 0. Where everything is
Handoff bucket (SPC scratch): `<SPC scratch bucket>`
```
code/        build_events.py make_landmarks.py qa_leakage.py make_splits.py common.py baselines.py eval.py (+ HANDOFF.md, PLAN.md)
data/        events.parquet landmarks.parquet splits.json genomics_status.pkl labels_raw.pkl top_genes.csv
results/     predictions_baselines_all.parquet results.csv (baseline metrics) + logs
```
Raw GENIE BPC (only needed to rebuild the event table): `<SPC raw-data bucket>` (about 297 MB; `aws s3 sync`). Rebuild is optional: `python build_events.py && python make_landmarks.py && python make_splits.py` takes about 1 minute. Do NOT regenerate splits.json unless told: it is fixed, and baseline predictions depend on it.

Setup: `pip install torch scikit-survival lifelines umap-learn pyarrow joblib`. Run scripts as `python script.py` from the project dir (do not use `python -I`, scripts import `common.py`). Never `pkill -f "<pattern that appears in your own command>"`; it kills your shell.

## 1. Data contract
**events.parquet** (one row per token; sorted by patient_id, day, rank): `patient_id, cohort, day (float, days from first index-cancer diagnosis; can be negative), rank (fixed within-day type order), etype, value, token (= etype:value), dclass (int bitmask: 1 chemo, 2 immuno, 4 targeted, 8 hormonal, 16 antiangiogenic; regimen tokens only), ref (regimen id "ca_seq_regimenno" for REG_START/REG_END)`.
- etypes: STATIC (COHORT=, HIST=, STAGE=, AGE=, SEX=; always 5 per patient, placed at the patient's earliest day), PANEL, SAMPLE, TMB, MUT, FUSION, AMP, DEL, PATH, TM, MET, MD, SCAN, REG_END, REG_START (rank order 0..14 in that order roughly; see `RANK` in build_events.py).
- Genomic tokens (PANEL/SAMPLE/TMB/MUT/FUSION/AMP/DEL) sit at the sequencing report day of their sample. Mutations have no own timestamp. Genes not on a panel are "untested": only positive findings plus a `PANEL:<assay>` token are in the sequence; the full mutated / wild-type / untested table is in `genomics_status.pkl` (baselines only).
- Counts: about 456k tokens, 7,394 patients. Median visible history at a landmark is 58 tokens (p90 131, p99 238, max 451).

**landmarks.parquet** (prediction points; 10,573 rows, 4,065 patients): `landmark_id, patient_id, cohort, institution, ref, landmark_day, first_seq_day, line, reg_name, dclass, STAGE, AGE, SEX, HIST, os_time (days), os_event, pfs_time, pfs_event (about 10% missing)`.
- A landmark is an index-cancer regimen start strictly after the patient's first sequencing report day (avoids left truncation; Brown et al. JAMA Oncol 2021). Labels (`os_*`, `pfs_*`) are TARGETS ONLY, never inputs. They were verified to match death / last-alive dates exactly.

**splits.json**: `{random5: [5 folds], loio: [DFCI, MSK, UHN, VICC], loco: [6 cohorts]}`; each fold has `name, train, test, pretrain` (patient id lists). `train`/`test` are patients with landmarks. `pretrain` = all patients (also ones without landmarks) except test patients and, for leave-one-out, the entire held-out institution/cohort. **Pretrain on `pretrain`, fine-tune on `train`, score `test`, per fold, from scratch. Test patients must never appear in pretraining.**

**History visible at a landmark** (use `common.prefix_index`, already verified: no event after the landmark day, last token is the landmark's own REG_START): events with (day, rank) strictly before (landmark_day, REG_START rank) plus the landmark's own REG_START. Other concurrent REG_START tokens on that day are not visible.

## 2. Baselines already done (point estimates, random5 pooled; test-fold predictions only)
KM by cohort C 0.567; Cox clinical 0.670; Cox + genomics 0.670; **GBSA on token counts 0.719 (AUC12 0.784, AUC24 0.827), the number to beat.** Leave-one-institution and leave-one-cohort results are in `results/results.csv` (final table with intervals once the evaluation finished; check the file's date). Expect lower numbers when cohorts are held out (KM by cohort scores exactly 0.5 inside a held-out cohort by construction).

## 3. Your tasks, in order
1. **Dataset/tokenizer (`hstu.py` or `data.py`).**
   - Vocabulary: tokens in the fold's `pretrain` events with count >= 5, others map to per-etype UNK. Embedding of value = per-token learned vector (UNK for rare).
   - Inputs per token: etype id, token id, dclass bitmask (5-d multi-hot), cohort id (with an extra UNKNOWN cohort id; during training replace the cohort by UNKNOWN with p=0.15 so held-out cohorts work), day (float).
   - Truncation to 256: keep the 5 STATIC tokens + ALL genomic tokens (PANEL/SAMPLE/TMB/MUT/FUSION/AMP/DEL) + the most recent remaining events; drop the oldest non-genomic events first. If genomic tokens alone exceed 200, keep the most recent samples.
   - Pretraining sequences: a patient's whole history, random window if longer than 256 (same static/genomic protection).
   - Merged-day ablation: all events on the same day (and same patient) collapsed to one token = sum/mean of member embeddings.
2. **Model (`hstu.py`)**, per the brief: embedding = type + value + drug-class (Linear on the 5-bit multi-hot) + cohort. Layer: `U,V,Q,K = split(SiLU(W1 x))`; `A = SiLU(Q K^T + pos_bias + time_bias) / N`, causal mask; output `x + W2(LayerNorm(A V) * U)`; dropout 0.2; hidden 64 to 128 (start 128), 2 to 4 layers (start 3), 4 heads.
   - `time_bias`: learned scalar per head per bucket of the gap in days between token i and j: `bucket = min(floor(log2(gap + 1)), 12)`; same-day = bucket 0. STATIC tokens use a fixed dedicated "static" bucket as key (they sit at diagnosis, far from late landmarks).
   - `pos_bias`: learned per head per relative distance (clipped at 255).
   - Switches: `--no_time_bias`, `--no_genomic` (drop genomic tokens from inputs), `--merge_day`, `--softmax_gpt` (Delphi-style: softmax attention + standard FFN block, no gating, same embeddings and time encoding as feasible), `--no_static_head`.
3. **Pretraining (`train.py`)**: next-event prediction at every position with three heads: event type (CE), value given type (CE over that type's vocabulary; use the type-masked logits), log-time bucket until the next event (CE over `floor(log2(gap+1))` buckets). Loss weights: type 1.0, value 1.0, time 0.5. **Down-weight the static opening tokens**: positions whose target is a STATIC token weight 0.1 (target static tokens are not predicted from nothing: weight 0 for the first position). AdamW, lr 3e-4 to 1e-3, warmup, early-stop on a 10% hold-out of the `pretrain` patients (never test patients). Per fold, per variant, from scratch.
4. **Fine-tuning**: discrete-time hazard head on the hidden state at the landmark REG_START token: `concat(h_landmark, static_emb)` -> MLP -> 60 logits (monthly bins of 30.44 days, up to 60 months). **`static_emb` = embedding of the landmark's 5 static tokens (mean of their final embeddings or a small separate embedding sum); this is a decided design choice because static tokens at the start of the sequence are far in time from late landmarks and may be down-weighted by the time bias and `/N`.** Loss for a patient with bin index k: event: `sum_{j<k} log(1-h_j) + log h_k`; censored: `sum_{j<=k-1} log(1-h_j)` (add the partial bin conservatively as not at risk); follow-up beyond 60 months is administratively censored in bin 60. Fine-tune on the fold's `train` landmarks with a 10% patient-level validation set for early stopping; choose the checkpoint by validation loss, not by test metrics.
   - Risk outputs per test landmark: `risk` = -expected survival time within 60 months (sum of S(t_k)); `risk12` = 1 - S(12 months); `risk24` = 1 - S(24 months). Because patients have several landmarks, shuffle by patient and sample all landmarks of a batch of patients.
5. **Prediction file contract** (so `eval.py` can score every model identically): a parquet with columns `landmark_id, split, fold, model, risk, risk12, risk24` (the last two may be NaN), only for that fold's TEST landmarks. Do not write training predictions. `eval.py <B> <pred files...>` -> `results.csv` (rows: model, split, fold or POOLED, metric in {cindex, auc12, auc24}, value, ci_lo, ci_hi). Patient-level bootstrap. If `risk12/risk24` are present, AUCs use them. (An updated `eval.py` with this feature is in `code/eval.py`; verify it reproduces the baseline numbers on `results/predictions_baselines_all.parquet` before using it.)
6. **Models to run** (names for the `model` column): `HSTU`, `GPT_delphi`, `HSTU_noTimeBias`, and the ablations `HSTU_noGenomic`, `HSTU_mergeDay`, `HSTU_noStaticHead`, plus the stretch `HSTU_textEmb`.
   - Priority by split: (1) random5 all models, (2) loio, (3) loco for the main three (`HSTU`, `GPT_delphi`, `HSTU_noTimeBias`); ablations on random5 only. If time is short, report fewer splits for HSTU and say so.
   - Use 200 bootstrap samples while developing, **1,000 for the final main results**.
   - First prove the pipeline end to end on NSCLC only (random5 folds restricted to NSCLC patients), then scale.
7. **Evaluation add-ons**: PFS as a secondary target (retrain the hazard head, or at least the baselines, with `pfs_time/pfs_event` on rows where available); first-line-only sensitivity (`line == 1` rows of the same predictions).
8. **Biology checks** (use the held-out predictions / models):
   - Cluster HSTU landmark embeddings (hidden state at landmark; from the fold model that did not train on that patient) and compare Kaplan-Meier curves within each cohort (log-rank test).
   - STK11 / KEAP1 mutation associated with OS in platinum-treated NSCLC: Cox / KM on the data, and show the model's risk shifts the same direction (compare mean predicted risk with/without the mutation within the same cohort and regimen class; also an in-silico test: flip the token and see the risk change).
   - TMB associated with OS in NSCLC on immunotherapy in second line or later (`line >= 2`, `dclass & 2`).
9. **Plots**: UMAP of embeddings colored by cohort and by subgroup; KM curves by subgroup; attention / time-bias visualization (learned time_bias per head vs. gap bucket). Everything into `plots/`.
10. **Stretch** (only after the core works): (a) LLM text embeddings of token strings from a LOCAL open-weights text encoder (no external API), projected and added to the learned value embedding, as ablation `HSTU_textEmb`, esp. for leave-one-cohort-out; (b) embedding-based discovery: clusters of landmark embeddings, enrichment of genes / regimens per cluster, KM and log-rank, stability across folds, hypothesis-generating only.
11. **Deliverables**: `events.parquet, landmarks.parquet, splits.json, hstu.py, train.py, baselines.py, eval.py, results.csv` (one row per model x split x metric with CIs), plots, a short `README.md` (data steps, leakage safeguards, how to rerun, honest limitations), and update `PLAN.md` change log for every change you make (UTC time, what, why).

## 4. Leakage safeguards you must preserve
- No outcome-derived column as an input: death / vital status / last contact, `os_*`, `pfs_*`, `tt_*`, anything computed from future events.
- Nothing after the landmark day is visible (use `common.prefix_index`; if you re-implement, assert it on all landmarks).
- Test patients absent from pretraining AND fine-tuning AND validation; hyperparameters chosen on inner validation only, never on test folds.
- Known small departures to document: the top-300 gene list (`top_genes.csv`) and the vocabulary frequency filter use all patients' feature frequencies (no labels).
- Four record_ids that appeared in two cohorts were dropped (could straddle a split).

## 5. Compute and process notes
- Check `torch.cuda.is_available()`. Models are small (about 0.5 to 2M parameters); one fold fits in minutes on a GPU. Run 15 folds x variants as a simple queue (1 to 3 processes per GPU). Seed everything; save a checkpoint and the test predictions per fold under `runs/<model>/<split>/<fold>/`.
- Save progress to `<SPC scratch bucket>` regularly (code, predictions, results, plots) so work survives a lost server.
- Sanity gates: a model with C-index > 0.80 on any split smells like leakage, stop and investigate; a model scoring clearly below the Cox clinical baseline (0.67) probably has a bug. Check the training loss falls, and that shuffling labels gives about 0.5.
- Suggested sanity check of GBSA: print its top features (expected legitimate signals: number of prior lines, progressing scans, metastasis sites).

## 6. Suggested timeline (PT)
- Sat evening: tokenizer, model, pretraining + hazard fine-tuning working on NSCLC random5; unit tests (causal mask, no future tokens, time-bias buckets).
- Sat night: all cohorts random5 for HSTU, GPT, no-time-bias; start loio / loco; ablations.
- Sun morning: final 1,000-sample CIs, biology checks, plots, README, demo assets. Freeze by Sun 1pm PT; leave buffer.
- Drop first if behind: tumor markers (`TM`) and pathology (`PATH`) tokens, then ablations, then loco for HSTU.
