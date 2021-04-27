import os
import csv
import spacy
import json
import tqdm
import numpy as np
import random
from transformers import AutoTokenizer
from wordfreq import zipf_frequency


HOMOGRAPHS = ["anchor", "arm", "band", "bank", "balance", "bar",
              "barrel", "bark", "bass", "bat", "battery", "beam", "board", "bolt",
              "boot", "bow", "brace", "break", "bug", "butt", "cabinet", "capital",
              "case", "cast", "chair", "change", "charge", "chest", "chip", "clip",
              "club", "cock", "counter", "crane", "cycle", "date", "deck", "drill",
              "drop", "fall", "fan", "file", "film", "flat", "fly", "gum", "hoe", "hood",
              "jam", "jumper", "lap", "lead", "letter", "lock", "mail", "match",
              "mine", "mint", "mold", "mole", "mortar", "move", "nail", "note",
              "offense", "organ", "pack", "palm", "pick", "pitch", "pitcher",
              "plaster", "plate", "plot", "pot", "present", "punch", "quarter",
              "race", "racket", "record", "ruler", "seal", "sewer", "scale", "snare",
              "spirit", "spot", "spring", "staff", "stock", "subject", "tank", "tear",
              "term", "tie", "toast", "trunk", "tube", "vacuum", "watch"]



def pos_tag(sentence):
    """POS tag a sentence with SpaCy without changing the number of tokens."""
    spacy_sent = nlp(sentence)
    tags = [w.pos_ for w in spacy_sent]
    tags_pruned = []
    for w in sentence.split():
        nlp_w = nlp(w)
        tags2 = [tags.pop(0) for w2 in nlp_w]
        if set(tags2) != {"PUNCT"} and "PUNCT" in tags2:
            tags2 = list(filter(("PUNCT").__ne__, tags2))
        tags_pruned.append(tags2[0])
    return tags_pruned


def tokenise(sentence, annotation, tags, freq_annotation):
    """Tokenise into MarianMT wordpieces."""
    tok_sent, tok_annotations, pos_tags, tok_freq_annotations = [], [], [], []
    for w, l, t, f in zip(sentence.split(), annotation, tags, freq_annotation):
        tokenised_w = tok.tokenize(w)
        tok_sent.extend(tokenised_w)
        tok_annotations.extend([l] * len(tokenised_w))
        tok_freq_annotations.extend([f] * len(tokenised_w))
        pos_tags.extend([t] * len(tokenised_w))
    return " ".join(tok_sent), tok_annotations, pos_tags, tok_freq_annotations


for language in ["de", "nl"]:
    for dataset in ["os18", "wmt19"]:
        nlp = spacy.load("en_core_web_sm")
        mname = f'Helsinki-NLP/opus-mt-en-{language}'
        tok = AutoTokenizer.from_pretrained(mname)

        data = open(f"data/{dataset}_wsd_bias_challenge_set.en", encoding="utf-8").readlines()
        for start, stop in [(0, 1000), (1000, 2000), (2000, 3000)]:
            samples, present = [], []
            for sentence in tqdm.tqdm(data[start:stop]):
                sentence = sentence.strip()
                annotation = []
                freq_annotation = []
                for word in sentence.split():
                    if any([x.lemma_ in HOMOGRAPHS for x in nlp(word)]):
                        annotation.append(1)
                        homograph = word
                    else:
                        annotation.append(0)
                    freq_annotation.append(zipf_frequency(word, "en"))
                present.append(1 in annotation)

                if 1 in annotation:
                    tags = pos_tag(sentence)
                    tokenised_sentence, tokenised_annotation, tokenised_tags, tokenised_freq_annotation = \
                        tokenise(sentence, annotation, tags, freq_annotation)
                    samples.append((
                        sentence,
                        ' '.join([str(x) for x in annotation]),
                        homograph,
                        ' '.join(tags),
                        tokenised_sentence,
                        ' '.join([str(x) for x in tokenised_annotation]),
                        ' '.join(tokenised_tags),
                        ' '.join([str(x) for x in tokenised_freq_annotation])))

            print(np.mean(present), len(present))

            with open(f"data/{dataset}_{start}-{stop}_{language}.tsv", 'w', encoding="utf-8") as f:
                for sample in samples:
                    f.write("\t".join(sample) + '\n')
