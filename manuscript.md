# Generative clinico-genomic timeline modelling for dynamic survival prediction across six cancers and four institutions

*Draft, 18 October 2026. All numbers come from the runs logged in PLAN.md. Citations are marked [ref] and need checking before
submission, as do the AACR GENIE BPC publication terms (cell-size rules, acknowledgement text).*

---

## Abstract

**Background.** Prognosis in advanced cancer changes as the disease progresses and treatment changes, but most prognostic
models are fitted once, at diagnosis, from a fixed set of clinical or genomic covariates. Generative sequence models of
health records can represent a whole patient history. Whether they improve treatment-time prognosis in oncology, and
whether they generalise to new institutions and cancer types, is not established.

**Methods.** We used AACR Project GENIE Biopharma Collaborative (BPC) data: 7,394 patients with non-small-cell lung,
colorectal, breast, pancreatic, prostate or bladder cancer, treated at four institutions. Each patient's curated history
became one time-stamped token sequence (455,020 events, 2,662 distinct tokens):
- diagnosis facts
- regimens, with drug-class bits
- imaging and clinician-assessed response
- metastatic sites
- tumour-marker levels
- pathology
- sequencing results: mutations, copy-number changes, fusions and tumour mutational burden (TMB)

We pretrained a Hierarchical Sequential Transduction Unit (HSTU) to predict the next event's type, value and time gap. We
then fine-tuned it with a discrete-time hazard head to predict overall survival (OS) at every regimen start after the first
sequencing report: 10,573 landmarks in 4,065 patients. We compared HSTU with:
- Kaplan–Meier by cohort
- Cox models with clinical and genomic covariates
- gradient-boosted survival trees on token counts (GBSA)
- Delphi, a generative health-trajectory transformer, retrained from scratch on the same data

Evaluation used three schemes: 5-fold patient-level cross-validation, leave-one-institution-out, and
leave-one-cancer-type-out. Differences were tested with a paired patient-level bootstrap.

**Results.**

| Pooled C-index | HSTU | GBSA | Delphi | Clinical Cox |
|---|---|---|---|---|
| Cross-validation | **0.743** [0.735–0.751] | 0.719 | 0.727 | 0.670 |
| Unseen institution | **0.745** | 0.714 | 0.714 | 0.665 |
| Unseen cancer type | **0.703** | 0.647 | not run | 0.632 |

- HSTU's lead over GBSA was +0.024 [0.016–0.033] in cross-validation, +0.031 [0.022–0.039] across institutions and
  +0.056 [0.047–0.064] across cancer types.
- HSTU was significantly better than GBSA in every held-out institution and every held-out cancer type.
- 12-month AUC was 0.805 for HSTU vs 0.784 for GBSA (cross-validation).
- Prespecified biology checks found that HSTU's risk responds strongly to stage. Single genomic tokens (STK11, KEAP1,
  TMB), however, moved predicted risk by less than 2 percentage points, and in some cases in the clinically wrong
  direction. This is despite clear associations with survival in the observed data (STK11 on platinum: adjusted HR 1.77).
- The patient embeddings, without supervision:
  - separate disease phases
  - recover treatment-defined subtypes (HER2-positive breast cancer with CDK12 co-amplification; a CDH1-enriched
    lobular cluster)
  - do not encode genomic alterations independently of the treatments those alterations lead to

**Conclusions.** One generative model, trained over a pooled pan-cancer, multi-institution clinico-genomic timeline, gives
treatment-time OS predictions that beat strong tabular and generative baselines, including in unseen institutions and
cancer types. In its current form, its gains come from clinical trajectory: response, metastatic spread and treatment
history. They do not come from tumour genomics, which it under-uses.

---

## 1. Introduction

Patients and oncologists ask about prognosis repeatedly: at diagnosis, and again at each progression and change of
therapy. Validated prognostic tools are mostly static, using stage, performance status and a few laboratory values at
diagnosis [ref]. Molecular profiling adds prognostic and predictive markers, but these are usually tested one at a time
[ref]. Real-world clinico-genomic datasets now record the whole course of care, from diagnosis and sequencing through
each line of therapy, imaging assessment, and death or censoring [ref GENIE BPC; ref MSK-CHORD]. A model that reads the
whole history and updates its prognosis at each decision point fits the clinical question more closely.

Transformer models of electronic health records (EHR) predict future diagnoses and events from token sequences of
patient history [refs BEHRT, Med-BERT, Foresight, ETHOS], and Delphi recently showed this at population scale for
multi-disease trajectories [ref Delphi]. In recommender systems, HSTU replaced softmax attention with a pointwise,
time-aware gated attention suited to long, irregular action sequences [ref HSTU]. Irregular timing and heterogeneous
event types are also typical of oncology care. To our knowledge, neither architecture has been evaluated for dynamic
survival prediction from curated pan-cancer clinico-genomic timelines, or tested for transfer to unseen institutions and
unseen cancer types.

**Contributions**
1. A unified tokenisation of curated clinical and genomic history across six cancers and four institutions, with a
   leakage-checked landmark design: prediction at each regimen start, using only information available up to that day.
2. HSTU pretraining with survival fine-tuning, which outperforms Cox, gradient-boosted and Delphi baselines in random,
   unseen-institution and unseen-cancer evaluation.
3. A prespecified biological audit using in-silico token edits with a positive control. It shows which inputs the model
   relies on, and reports a negative result for single genomic markers.
4. An analysis of token and patient embeddings, including a tokenisation artefact that explains part of the weak use of
   genomics.
5. An interactive what-if simulator for prognosis at any treatment step (internal use only).

## 2. Related work

**Static and dynamic prognostic models.**
- Cox regression with clinical covariates is standard. Landmarking extends it to prediction from later time points
  [ref van Houwelingen].
- Machine-learning survival models include random survival forests, gradient-boosted Cox models [ref sksurv/GBSA],
  DeepSurv and discrete-time neural hazard models [refs DeepSurv; Nnet-survival; DeepHit].
- Our survival head is a discrete-time hazard model with monthly bins.

**Clinico-genomic real-world data.**
- Linked records of genomics and clinical outcomes have been used for prognosis and biomarker discovery, for example in
  the Flatiron–Foundation Medicine database and MSK-CHORD [refs].
- GENIE BPC provides PRISSMM-curated regimens, imaging and pathology for sequenced patients at several institutions
  [refs GENIE BPC; PRISSMM].
- Most prior models use per-patient tabular summaries rather than the event sequence.

**Sequence models of health records.**
- BEHRT, Med-BERT and CEHR-BERT are masked-token encoders [refs].
- Foresight, ETHOS and Delphi are autoregressive generators [refs]. Delphi models disease onset and death with
  next-token and waiting-time losses, and reads risk directly from predicted event rates.
- HSTU comes from large-scale recommendation [ref]. It replaces softmax attention with SiLU-gated attention plus learned
  relative-position and time biases.
- We compare HSTU with Delphi as published (generative only, no survival fine-tuning) and with tabular baselines.

**Oncology biomarkers used for audit.**
- STK11 and KEAP1 alterations are associated with poor outcome in NSCLC, particularly with immunotherapy and
  chemo-immunotherapy [refs Skoulidis 2018; others].
- High TMB is associated with benefit from immune-checkpoint blockade [refs].
- We use these known associations as biological checks on the model, not as new findings.

## 3. Data

**Source.** AACR Project GENIE BPC, six cohorts (NSCLC, CRC, breast, pancreatic, prostate, bladder) from DFCI, MSK, UHN
and VICC. Analyses ran inside the SPC AWS environment. No data left it, and no record linkage or re-identification was
attempted.

**Patients**

| Cohort | DFCI | MSK | UHN | VICC | Patients with landmarks | Landmarks | Deaths* | Median line† |
|---|---|---|---|---|---|---|---|---|
| NSCLC | 273 | 440 | 32 | 133 | 878 | 1,873 | 64% | 3 |
| CRC | 273 | 429 | 0 | 130 | 832 | 2,257 | 56% | 4 |
| Breast | 256 | 379 | 0 | 115 | 750 | 2,523 | 59% | 6 |
| Pancreatic | 216 | 343 | 34 | 35 | 628 | 1,331 | 82% | 3 |
| Prostate | 189 | 378 | 45 | 20 | 632 | 1,858 | 50% | 6 |
| Bladder | 142 | 133 | 27 | 43 | 345 | 731 | 62% | 3 |
| **All** | 1,349 | 2,102 | 138 | 476 | **4,065** | **10,573** | 61% | 4 |

\*Share of landmarks with an observed death. †Regimen index, counting regimens before sequencing.

- From each patient's first eligible landmark: median OS 22.3 months; median follow-up (reverse Kaplan–Meier) 36.7 months.
- A further 3,329 patients have no eligible landmark. They are used only for pretraining, giving 7,394 sequences in all.

**Timeline construction** (`build_events.py`)
- Day 0 is the first diagnosis of the index cancer. Each event becomes a token `TYPE:value`.
- **Diagnosis facts (STATIC):** cohort, histology, stage, age band, sex.
- **Treatment:** start and end of each regimen, named by its sorted drug list, plus a 5-bit drug class (chemotherapy,
  immunotherapy, targeted, hormonal, anti-angiogenic).
- **Disease status:**
  - imaging response (SCAN) and clinician-assessed status (MD): progressing, stable, improving, mixed, indeterminate
  - first date at each metastatic site (MET, 9 sites)
  - tumour markers as ratio-to-upper-normal buckets (TM)
  - pathology procedure type
- **Genomics, dated at the sequencing report:**
  - non-silent mutations (top 300 genes by patient frequency)
  - copy-number ±2 as AMP/DEL
  - somatic non-intragenic fusions
  - TMB, bucketed
  - panel and sample type
- Reports after death are excluded.
- Within a day, tokens follow a fixed order: diagnosis facts and genomics, then pathology, markers, metastases,
  assessments, imaging, regimen ends, and regimen starts last.
- Size: 455,020 events and 2,662 distinct tokens. Median 48 tokens per patient; 0.5% exceed the 256-token context.

**Landmarks and outcome**
- A landmark is any regimen start on or after the first sequencing report.
- Outcome: OS from the landmark day (death, or censoring at last known alive).
- The model input is everything up to and including the landmark's own REG_START, which is the final token.
- An automated leakage check (`qa_leakage.py`) asserts that no input event is dated after the landmark day.

**Splits** (`make_splits.py`), all at patient level
- random5: 5-fold cross-validation, about 813 test patients per fold.
- loio: leave-one-institution-out (4 folds).
- loco: leave-one-cancer-type-out (6 folds).
- In each fold, pretraining uses every non-test patient. Survival fitting, for all models, uses only train patients with
  landmarks.

## 4. Methods

### 4.1 HSTU

**Input embedding.** Sum of:
- token embedding
- event-type embedding
- linear projection of the drug-class bits
- cohort embedding
- embedding of log-scaled days since diagnosis

**Layers.** Each layer computes (U, V, Q, K) = split(SiLU(W·LN(x))) and attention
A = SiLU(QKᵀ + b_pos + b_time) / N, with causal masking. Output = W_o·(LN(A·V) ⊙ U), with a residual connection.
- b_pos is a learned relative-position bias.
- b_time is a learned bias per head for each log₂(gap+1) bucket of the day gap between query and key. Static tokens get
  their own bucket.
- Size: d = 128, 3 layers, 4 heads, dropout 0.2, context 256. When truncating, diagnosis facts and the most recent
  genomic tokens are always kept.

**Vocabulary.** Built per fold from pretraining events: tokens seen 5 or more times (about 1,216). Rarer tokens map to an
unknown token for their event type.

**Pretraining.** Next-event prediction on all non-test patients:
L = CE(type) + CE(value | type) + 0.5·CE(log₂ gap bucket)
- The value softmax is restricted to the true event type.
- Static targets are down-weighted to 0.1.
- AdamW at 1e-3, one-cycle schedule, up to 40 epochs, early stopping on a 10% patient hold-out (patience 4).

**Survival fine-tuning.**
- Representation z = [h at landmark token; mean h over static tokens].
- Head: MLP(256→128→60), giving monthly hazards h_j = σ(ℓ_j) and survival S(m) = ∏_{j≤m}(1−h_j).
- Loss: discrete-time negative log-likelihood. Death in month k contributes Σ_{j<k}log(1−h_j) + log h_k; censoring at k
  contributes Σ_{j<k}log(1−h_j). Follow-up beyond 60 months is censored at 60.
- All weights are trained: AdamW, encoder learning rate 3e-4, head 9e-4. Each batch holds whole patients.
- 15% cohort-ID dropout, so the model can score an unseen cancer type: the held-out cohort gets the unknown ID in loco.
- The number of epochs is chosen on a 10% inner validation set of train patients. The model is then refit from the
  pretrained weights on all train patients for that many epochs.
- Risk score for the C-index: −Σ_m S(m) (negative expected months survived within 5 years). 1−S(12) and 1−S(24) give the
  horizon AUCs.

### 4.2 Baselines (`baselines.py`)
1. **KM by cohort:** risk = −restricted mean survival to 3 years in the landmark's cohort.
2. **Cox clinical** (ridge-penalised, penalty chosen by grouped CV on train patients): cohort and stage one-hots, age
   band (ordinal), log line number, regimen drug-class bits, sex.
3. **Cox clinical + genomics:** adds 0/1 alteration flags for genes covered by every panel, as of the landmark.
4. **GBSA:** gradient-boosted Cox trees (150 trees, depth 3, learning rate 0.05, subsample 0.7, ≥20 per leaf, √p
   features). Inputs are counts of 671 tokens (frequency ≥30) in the history up to the landmark. It has no order or
   timing information.
5. **Delphi** [ref]: the published architecture (vendored, MIT licence), retrained from scratch in each fold on the same
   token sequences.
   - Pretraining patients who died get a DEATH token on the day of death. "No event" tokens are added roughly every 180
     days up to death or censoring.
   - Loss: next-token cross-entropy plus an exponential waiting-time loss.
   - Early stopping uses inner-validation C-index (patience 10), the same rule as HSTU.
   - Risk = predicted DEATH rate at the landmark token. As published, there is no survival fine-tuning.

### 4.3 Evaluation (`eval.py`, `paired_diff.py`)
- Harrell's C-index for OS, and time-dependent cumulative/dynamic AUC at 12 and 24 months (IPCW, censoring distribution
  from training landmarks).
- Metrics are pooled over all test landmarks of a split (each landmark scored by its own fold's model) and also reported
  per fold.
- 95% CIs come from a patient-level bootstrap (B = 200 for single models, B = 1,000 for paired differences). Paired
  tests resample patients jointly for both models.

**Controls**
- Shuffled-label fine-tuning (fold 0) checks that the pipeline leaks nothing.
- Three seeds (random5) check run-to-run stability.
- An HSTU variant without the time bias tests that component.

### 4.4 Biological audit (`biology.py`, `simulate.py`)
Specified before looking at the flip results:
1. **Observed associations** at each patient's first landmark: STK11/KEAP1 vs wild-type on platinum, TMB-high vs low on
   an immunotherapy line ≥2 vs non-immunotherapy lines. Cox models adjusted for line and stage.
2. **Model view:** out-of-fold HSTU 12-month risk in the same groups.
3. **In-silico flips:** each test patient's prefix is edited, scored by that fold's model, and the change in 12-month risk
   is bootstrapped over patients (B = 1,000). Edits: add or remove MUT:STK11 or MUT:KEAP1 (removal also strips
   deletions); set TMB to >6 vs 0–0.5; stage IV vs I as a positive control.

### 4.5 Embedding analysis (`analysis/emb_*.py`)
- **Token level:**
  - raw input embeddings
  - contextual embeddings (each token's mean final hidden state over its occurrences)
  - stability across fold models, nearest neighbours, regimen and gene structure
- **Patient level:**
  - representation z at each patient's first landmark, computed by the fold model for which the patient was a test
    patient
  - fold spaces aligned to fold 0 by orthogonal Procrustes
  - within each cohort: linear probes (5-fold CV AUC) for marker status, and k-means clustering (k = 2–6 by silhouette)
    before and after regressing out predicted risk and line
  - clusters characterised by token enrichment and Kaplan–Meier

## 5. Results

### 5.1 Discrimination

**Table 2. Pooled test performance (95% bootstrap CI).**

| Split | Model | C-index | AUC 12 mo | AUC 24 mo |
|---|---|---|---|---|
| random5 | KM cohort | 0.567 | 0.581 | 0.590 |
| | Cox clinical | 0.670 [0.660–0.680] | 0.725 | 0.772 |
| | Cox clinical + genomics | 0.670 [0.659–0.681] | 0.726 | 0.770 |
| | GBSA counts | 0.719 [0.709–0.728] | 0.784 | 0.827 |
| | Delphi | 0.727 [0.718–0.736] | 0.783 | 0.808 |
| | **HSTU** | **0.743 [0.735–0.751]** | **0.805** [0.794–0.816] | **0.837** [0.826–0.852] |
| loio | Cox clinical | 0.665 [0.655–0.674] | 0.719 | 0.763 |
| | GBSA counts | 0.714 [0.705–0.722] | 0.779 | 0.821 |
| | Delphi | 0.714 [0.705–0.723] | 0.767 | 0.794 |
| | **HSTU** | **0.745 [0.736–0.754]** | **0.806** | **0.840** |
| loco | Cox clinical | 0.632 [0.621–0.642] | 0.685 | 0.730 |
| | GBSA counts | 0.647 [0.636–0.657] | 0.697 | 0.756 |
| | **HSTU** | **0.703 [0.694–0.711]** | **0.762** | **0.797** |

Pooled C-indices mix the fold models, so they differ slightly from the per-fold means: random5 HSTU 0.760 vs GBSA 0.720.
Adding panel-wide genomic flags to the clinical Cox model did not improve it.

**Table 3. Paired difference in C-index, HSTU minus comparator (patient bootstrap B = 1,000).**

| Split | Held-out group | vs GBSA | vs Delphi |
|---|---|---|---|
| random5 | pooled | +0.024 [0.016, 0.033] | +0.016 [0.008, 0.023] |
| | per fold | | +0.013 to +0.036; 4 of 5 folds exclude 0 |
| loio | DFCI (3,381 landmarks) | +0.031 [0.018, 0.044] | +0.024 [0.011, 0.037] |
| | MSK (5,732) | +0.017 [0.005, 0.029] | +0.026 [0.014, 0.037] |
| | UHN (316) | +0.068 [0.024, 0.114] | +0.020 [−0.021, 0.063] |
| | VICC (1,144) | +0.081 [0.053, 0.107] | +0.021 [−0.001, 0.045] |
| | pooled | +0.031 [0.022, 0.039] | +0.031 [0.023, 0.038] |
| loco | Bladder (731) | +0.043 [0.008, 0.078] | — |
| | Breast (2,523) | +0.058 [0.037, 0.081] | — |
| | CRC (2,257) | +0.055 [0.033, 0.075] | — |
| | NSCLC (1,873) | +0.034 [0.015, 0.054] | — |
| | Pancreatic (1,331) | +0.057 [0.039, 0.074] | — |
| | Prostate (1,858) | +0.059 [0.041, 0.075] | — |
| | pooled | +0.056 [0.047, 0.064] | — |

**Per-fold C-index in the unseen-institution and unseen-cancer splits**

| Held-out | HSTU | GBSA | Delphi | Cox |
|---|---|---|---|---|
| DFCI | 0.751 | 0.721 | 0.728 | 0.663 |
| MSK | 0.753 | 0.736 | 0.727 | 0.686 |
| UHN | 0.679 | 0.610 | 0.659 | 0.594 |
| VICC | 0.701 | 0.620 | 0.679 | 0.592 |
| Bladder | 0.718 | 0.675 | — | 0.580 |
| Breast | 0.752 | 0.694 | — | 0.671 |
| CRC | 0.689 | 0.634 | — | 0.576 |
| NSCLC | 0.683 | 0.649 | — | 0.626 |
| Pancreatic | 0.728 | 0.671 | — | 0.627 |
| Prostate | 0.811 | 0.752 | — | 0.747 |

HSTU's margin over GBSA grows as the test distribution moves further from training: +0.024 (random), +0.031 (unseen
institution), +0.056 (unseen cancer). It is largest for the two smaller institutions, UHN and VICC, whose sequencing
panels are not seen in training.

**Controls**
- **Shuffled labels:** C = 0.468 (fold 0).
- **Seeds:** mean per-fold C was 0.760, 0.762 and 0.763 for seeds 0–2 (fold SD 0.010–0.015).
- **Time bias:** removing it did not reduce performance. Random5 per-fold mean was 0.760 vs 0.760; pooled paired
  difference −0.005 [−0.009, −0.001], favouring the variant without it; loio mean 0.719 vs 0.721.

### 5.2 Biological audit

**Table 4. Observed outcomes versus model behaviour.**

| Check | Observed data | HSTU |
|---|---|---|
| STK11-mutant vs wild-type, NSCLC on platinum | adjusted HR 1.77 [1.20, 2.60] | out-of-fold 12-mo risk 0.31 vs 0.35 (wrong direction) |
| STK11 or KEAP1 vs neither, platinum | adjusted HR 1.79 | — |
| Add/remove STK11 or KEAP1 token | — | Δ risk < 0.6 percentage points, sign inconsistent |
| TMB >6 vs 0–0.5, IO line ≥2 | HR 0.90 (n.s.) | Δ risk +1.5 pp, the same with and without IO |
| TMB high vs low, non-IO line | HR 1.48 | — |
| **Stage IV vs I (positive control)** | — | **Δ risk +5.8 pp [5.5, 6.0]; positive in 98% of patients** |

The model responds strongly and consistently to clinical inputs but hardly at all to single genomic tokens, including
markers with clear associations in the observed data.

The simulator also shows strong, sensible responses to disease-status inputs. In one NSCLC example, adding new liver
and brain metastases at first line raised predicted 12-month risk from 24% to 47%. Predicted risk rose from 24% to 89%
over successive lines.

### 5.3 What the embeddings encode

**Tokens**
- Raw input embeddings stayed near their random starting values: row norm 11.4 ≈ √128, and similarity between tokens
  agreed little across folds (Spearman 0.02–0.05).
- Contextual embeddings were stable across folds (Spearman 0.85–0.89), and 98% of each token's 5 nearest neighbours had
  the same event type.
- They recovered pharmacological and clinical groupings without supervision:
  - pembrolizumab ≈ atezolizumab ≈ nivolumab
  - osimertinib ≈ alectinib ≈ crizotinib ≈ erlotinib
  - leuprolide ≈ degarelix ≈ triptorelin ≈ goserelin
  - stages I–IV with each other
  - progressing clinician assessment ≈ progressing imaging
- Regimens grouped more by cancer type (mean cosine 0.57 vs 0.39) than by drug class (0.49 vs 0.41).
- **Mutation tokens were ordered alphabetically,** not by biology: EGFR → ERBB2, EPHA3, ETV6; STK11 → STAG2, TERT,
  TET1. Similarity did not track co-mutation frequency (Spearman 0.04). The cause is a tokenisation artefact: same-day
  genomic tokens were sorted by name (100% of 4,956 multi-mutation reports), so next-token pretraining learned to
  predict alphabetical succession. The artefact cannot leak outcome information; it only weakens genomic
  representation learning.

**Patients**
- Cancer type explained 57% of the variance in the top 20 principal components. The first component tracked predicted
  risk (ρ = −0.58).
- **Clustering within each cohort separated disease phases**, with large survival differences (log-rank p from 10⁻¹⁸ to
  10⁻⁶¹):
  - breast: early, curative-intent vs heavily pretreated metastatic (median OS not reached vs 21 months)
  - prostate: hormone-sensitive vs post-taxane castration-resistant (not reached vs 15 months)
  - bladder: non-muscle-invasive on BCG vs progressing on chemotherapy (not reached vs 9 months)
- In NSCLC and pancreatic cancer the clusters had no prognostic separation.
- **Linear probes:**
  - Status tied to a treatment or histology was recovered well: squamous histology 0.99, prior immunotherapy 0.94, CRC
    TMB >6 0.89, ERBB2 amplification 0.88, ALK fusion 0.82, EGFR 0.81.
  - Alterations without a matched therapy were not: PIK3CA, APC, SMAD4, TMPRSS2, FGFR3 and KRAS in CRC/pancreatic all
    0.56–0.64.
  - EGFR probe AUC fell from 0.81 to 0.65 among TKI-naive patients, so the model mostly knows EGFR status through
    exposure to EGFR inhibitors.
- **After regressing out predicted risk and line**, breast cancer split into:
  - a HER2-positive cluster: pertuzumab + trastuzumab; CDK12 amplification 15% vs 5% (CDK12 is co-amplified with ERBB2
    at 17q12)
  - a CDH1-enriched (29% vs 9%) lobular/hormone-receptor-positive cluster with letrozole + palbociclib exposure
  - trial-predominant and later-line-chemotherapy clusters
  - All with significantly different survival (log-rank p = 3×10⁻⁴).
- One small CRC cluster was enriched for one institution's sequencing panel, a site signal entering through panel tokens.

## 6. Discussion

**Main findings**
- Pretraining a generative model on pooled pan-cancer timelines, then fine-tuning it for landmark survival, gave the best
  discrimination of every method tested, in every evaluation scheme.
- The gain over GBSA on the same tokens was largest in the hardest settings: a new institution, especially a small one,
  and a new cancer type. This suggests the model learns disease-course patterns that transfer, such as response
  trajectories, metastatic spread and treatment sequence, rather than site- or cancer-specific frequencies.

**What drives the predictions**
- The audit and the embedding analysis agree: the model's signal is clinical trajectory, not tumour genomics.
- Single genomic tokens hardly move risk. Genomic status is represented mostly through the treatments it leads to.
- Two causes are identified:
  1. the alphabetical within-report ordering, which turns part of genomic pretraining into an alphabet task
  2. genomic tokens are sparse and weak per patient relative to the many repeated assessment tokens
- In line with this, adding genomics to the Cox model did not help either. Genomics' marginal prognostic value at later
  landmarks may simply be small once treatment and response history are known.

**Architecture vs objective.**
- HSTU outperformed Delphi by 0.016 (random) and 0.031 (unseen institution).
- This comparison is faithful to Delphi as published, but it confounds architecture with training objective: HSTU is
  fine-tuned on the landmark survival task, while Delphi scores risk from its generative death rate.
- HSTU's time-aware attention bias added nothing here (the day-since-diagnosis embedding may already capture timing).
- So we do not attribute the gain to the HSTU attention design. A standard softmax transformer with the same
  pretraining and fine-tuning is the necessary ablation.

**Limitations**
1. Curated BPC data from academic centres; two institutions contribute 85% of patients, and UHN is small (316
   landmarks).
2. OS only, no calibration analysis yet, and no performance-status variable, which is a major clinical prognostic
   factor.
3. Landmarks from the same patient are correlated. CIs use a patient-level bootstrap, but the training loss weights
   patients with many lines more heavily.
4. Baselines lack explicit timing or recency features, so part of HSTU's advantage over GBSA may reflect access to timing
   rather than to sequence modelling.
5. Within-day ordering artefact for genomic tokens (Section 5.3).
6. Sequencing panel tokens identify the institution; in loio the held-out site's panels are unseen.
7. Delphi was not evaluated on loco, and the HSTU vs Delphi comparison is confounded by objective.
8. Biology flips change one token in an otherwise unchanged history. They test model reliance, not causal effects.

**Planned work**
- Ablations: a softmax transformer with identical fine-tuning; no pretraining; no genomic tokens; no panel tokens.
- Shuffling same-day genomic tokens in pretraining, with the biology audit repeated.
- A GBSA baseline with timing and recency features.
- Calibration (integrated Brier score, calibration at 12 and 24 months) and analysis by line of therapy.
- External validation on a non-BPC clinico-genomic cohort.

## 7. Conclusion

A single HSTU model over unified clinico-genomic timelines predicts overall survival at each regimen start better than
Cox, gradient-boosted and Delphi baselines. Its margin is largest at unseen institutions and in unseen cancer types.
Prespecified audits show the model relies on disease trajectory and largely ignores tumour genomics, partly because of a
correctable tokenisation artefact. Treatment-time generative prognosis looks promising, but its use of molecular data
needs further work before clinical claims.

---

## Data and code availability

- **Data:** AACR Project GENIE BPC data are available to approved users under the GENIE data-use agreement. No data,
  derived tables, predictions or weights are distributed.
- **Code:** github.com/yzzhong19/PatientFoundationModel. The Delphi model code is vendored under its MIT licence.

## Acknowledgements
[AACR Project GENIE BPC acknowledgement text, as required by the data-use terms.]

## References (to verify)
- Zhai J, et al. Actions speak louder than words: trillion-parameter sequential transducers for generative
  recommendations (HSTU). ICML 2024.
- Shmatko A, et al. Learning the natural history of human disease with generative transformers (Delphi). Nature 2025.
- Li Y, et al. BEHRT: transformer for electronic health records. Sci Rep 2020.
- Rasmy L, et al. Med-BERT. NPJ Digit Med 2021.
- Kraljevic Z, et al. Foresight: a generative pretrained transformer for modelling patient timelines. Lancet Digit Health 2024.
- Renc P, et al. Zero-shot health trajectory prediction using transformer (ETHOS). NPJ Digit Med 2024.
- van Houwelingen HC. Dynamic prediction by landmarking in event history analysis. Scand J Stat 2007.
- Gensheimer MF, Narasimhan B. A scalable discrete-time survival model for neural networks. PeerJ 2019.
- Katzman JL, et al. DeepSurv. BMC Med Res Methodol 2018. Lee C, et al. DeepHit. AAAI 2018.
- Pölsterl S. scikit-survival. JMLR 2020.
- AACR Project GENIE Consortium. Cancer Discov 2017; GENIE BPC cohort papers (NSCLC, CRC, and others).
- Kehl KL, et al. PRISSMM curation framework.
- Jee J, et al. Automated real-world data integration improves cancer outcome prediction (MSK-CHORD). Nature 2024.
- Singal G, et al. Clinico-genomic database. JAMA 2019.
- Skoulidis F, et al. STK11/LKB1 mutations and PD-1 inhibitor resistance in KRAS-mutant lung adenocarcinoma. Cancer
  Discov 2018.
