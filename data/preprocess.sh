#!/bin/bash

python preprocess_json.py --pos_tag_method spacy --output_filename idiom_annotations_0.5.tsv --frequency_threshold 0.5
wait


#python translate_keywords.py --input_tsv idiom_annotations.tsv --output_tsv idiom_annotations_translated.tsv