#!/bin/bash

python preprocess_json.py --pos_tag_method spacy --output_filename idiom_keywords.tsv --frequency_threshold 0.5
wait

python translate_keywords.py --input_tsv idiom_keywords.tsv --output_tsv idiom_keywords_translated.tsv
wait

for i in {0..1726}
do
   python translate_magpie.py --source corpus/magpie/${i}.tsv --pred ${i}_pred.txt --folder corpus
   wait
done