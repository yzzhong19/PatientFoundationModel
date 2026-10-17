#!/bin/bash
until grep -q QUEUE_FINISHED queue_hstu_random5.log; do sleep 15; done
python train.py --model HSTU_shuffledLabels --split random5 --fold fold0 --shuffle_labels > shuffle_fold0.log 2>&1
python -c "
import pandas as pd
p=pd.read_parquet('predictions_baselines_all.parquet'); p=p[p.split=='random5'].drop(columns=['note'],errors='ignore'); p.to_parquet('pred_baselines_random5.parquet')"
python eval.py 200 runs/HSTU/random5/*/predictions.parquet runs/HSTU_shuffledLabels/random5/fold0/predictions.parquet pred_baselines_random5.parquet > eval_random5_hstu.log 2>&1
mv results.csv results_random5_hstu_B200.csv; cp results_B200_baselines.csv results.csv
python paired_diff.py 1000 HSTU GBSA_counts random5 runs/HSTU/random5/*/predictions.parquet pred_baselines_random5.parquet > paired_random5.log 2>&1
echo ALL_DONE >> eval_random5_hstu.log
