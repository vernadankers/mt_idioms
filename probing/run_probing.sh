#!/bin/bash

python train_probes.py --label_type magpie --n 0 --m 1727 --average_over_pie
wait
python train_probes.py --label_type both --n 0 --m 1727 --average_over_pie
wait