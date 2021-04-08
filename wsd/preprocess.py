import os
import csv
import spacy
import json
import tqdm
import numpy as np
import random
from transformers import AutoTokenizer


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


def tokenise(sentence, annotation, tags):
    """Tokenise into MarianMT wordpieces."""
    tok_sent, tok_annotations, pos_tags = [], [], []
    for w, l, t in zip(sentence.split(), annotation, tags):
        tokenised_w = tok.tokenize(w)
        tok_sent.extend(tokenised_w)
        tok_annotations.extend([l] * len(tokenised_w))
        pos_tags.extend([t] * len(tokenised_w))
    return " ".join(tok_sent), tok_annotations, pos_tags


nlp = spacy.load("en_core_web_sm")
mname = f'Helsinki-NLP/opus-mt-en-de'
tok = AutoTokenizer.from_pretrained(mname)
data = open("data/wmt19_wsd_bias_challenge_set.en", encoding="utf-8").readlines()


homographs = ["anchor", "arm", "band", "bank", "balance", "bar",
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


for start, stop in [(0, 1000), (1000, 2000), (2000, 3000)]:
    samples, present = [], []
    for sentence in tqdm.tqdm(data[start:stop]):
        sentence = sentence.strip()
        annotation = []
        for word in sentence.split():
            if word in homographs:
                annotation.append(1)
                homograph = word
            else:
                annotation.append(0)
        present.append(1 in annotation)

        if 1 in annotation:
            tags = pos_tag(sentence)
            tokenised_sentence, tokenised_annotation, tokenised_tags = tokenise(sentence, annotation, tags)
            samples.append((
                sentence,
                ' '.join([str(x) for x in annotation]),
                homograph,
                ' '.join(tags),
                tokenised_sentence,
                ' '.join([str(x) for x in tokenised_annotation]),
                ' '.join(tokenised_tags)))

    print(np.mean(present), len(present))

    with open(f"wmt19_{start}-{stop}.tsv", 'w', encoding="utf-8") as f:
        for sample in samples:
            f.write("\t".join(sample) + '\n')


# The boy pulled himself together so hastily that his limbs got in the way , and his wood violin was within a hair 's breadth of falling into the fire .
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
# a hair's breadth
# figurative
# misc
# DET NOUN VERB PRON ADV ADV ADV SCONJ DET NOUN VERB ADP DET NOUN PUNCT CCONJ DET NOUN NOUN AUX ADP DET NOUN PART NOUN ADP VERB ADP DET NOUN PUNCT
# det nsubj ROOT dobj advmod advmod advmod mark poss nsubj ccomp prep det pobj punct cc poss compound nsubj conj prep det poss case pobj prep pcomp prep det pobj punct
# 1 2 2 2 2 6 2 10 9 10 6 10 13 11 10 2 18 18 19 2 19 22 24 22 20 24 25 26 29 27 19
# ▁The ▁boy ▁pulled ▁himself ▁together ▁so ▁has ti ly ▁that ▁his ▁limbs ▁got ▁in ▁the ▁way ▁ , ▁and ▁his ▁wood ▁ viol in ▁was ▁within ▁a ▁hair ▁' s ▁breadth ▁of ▁falling ▁into ▁the ▁fire ▁ .
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
# DET NOUN VERB PRON ADV ADV ADV ADV ADV SCONJ DET NOUN VERB ADP DET NOUN PUNCT PUNCT CCONJ DET NOUN NOUN NOUN NOUN AUX ADP DET NOUN PART PART NOUN ADP VERB ADP DET NOUN PUNCT PUNCT