#!/bin/bash

python svcca_influence.py  --n 0 --m 1727 --k 1 --label 0 --masked_folder ../data/magpie/masked_idiom --neighbourhood 10 --context_context --save_file "data/svcca_idi2con.pickle"
wait

python svcca_influence.py  --n 0 --m 1727 --k 1 --label 1 --masked_folder ../data/magpie/masked_idiom --neighbourhood 10 --context_context --save_file "data/svcca_idi2idi.pickle"
wait

python svcca_influence.py  --n 0 --m 1727 --k 1 --label 1 --masked_folder ../data/magpie/masked_context --neighbourhood 10 --context_context --save_file "data/svcca_con2idi.pickle"
wait

python svcca_influence.py  --n 0 --m 1727 --k 1 --label 0 --masked_folder ../data/magpie/masked_context --neighbourhood 10 --context_context --save_file "data/svcca_con2con.pickle"
wait

python appendix.py
wait

python over_layers.py
wait
