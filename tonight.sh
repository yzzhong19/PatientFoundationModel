#!/bin/bash
# wait for the running Delphi fold0 (writes runs/Delphi/random5/fold0) then run tonight's queue, 2 in parallel
until [ -z "$(pgrep -f '[d]elphi_train.py --split random5 --fold fold0')" ]; do sleep 20; done
./queue.sh jobs_tonight.txt 2
