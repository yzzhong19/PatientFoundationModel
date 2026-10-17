# PatientFoundationModel

HSTU generative sequence model over cancer patient timelines (AACR GENIE BPC), predicting overall survival at each regimen
start, compared with Kaplan-Meier, Cox, GBSA and a retrained Delphi.

**Data is not included.** GENIE BPC is controlled-access; all data, derived tables, predictions and model weights stay in the
SPC AWS environment. This repository holds code only.

Pipeline: `build_events.py` -> `make_landmarks.py` -> `qa_leakage.py` -> `make_splits.py` -> `baselines.py` / `train.py` (HSTU) /
`delphi_train.py` -> `eval.py`, `paired_diff.py` -> `biology.py`; `dashboard.py` is an interactive what-if simulator (Dash).
Batch runs: `./queue.sh jobs_hstu.txt 1`. Tests: `python test_hstu.py`.

`delphi_model.py` is vendored unchanged from github.com/gerstung-lab/Delphi (MIT, see `LICENSE-delphi`).
