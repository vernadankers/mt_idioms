#!/bin/bash

python preprocess.py
wait

for lang in en nl 
do
    python translate_wsd.py --source data/os18_0-1000_${lang}.tsv --pred data/os18_0-1000_${lang}_pred.txt --lang $lang
    wait
    python translate_wsd.py --source data/os18_1000-2000_${lang}.tsv --pred data/os18_1000-2000_${lang}_pred.txt --lang $lang
    wait
    python translate_wsd.py --source data/os18_2000-3000_${lang}.tsv --pred data/os18_2000-3000_${lang}_pred.txt --lang $lang
    wait
    python translate_wsd.py --source data/wmt19_0-1000_${lang}.tsv --pred data/wmt19_0-1000_${lang}_pred.txt --lang $lang
    wait
    python translate_wsd.py --source data/wmt19_1000-2000_${lang}.tsv --pred data/wmt19_1000-2000_${lang}_pred.txt --lang $lang
    wait
    python translate_wsd.py --source data/wmt19_2000-3000_${lang}.tsv --pred data/wmt19_2000-3000_${lang}_pred.txt --lang $lang
    wait
done