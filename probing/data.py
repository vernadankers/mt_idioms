import pickle
import os
import numpy as np
import torch
import copy
import random
import argparse
import seaborn as sns
from classifier import Classifier
from collections import defaultdict, Counter
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
from l0module import L0Linear, HardConcreteLinear
import os
import random
import logging
import argparse
import torch
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    logging.info(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Sample():
    def __init__(self, word, vector, translation_label, magpie_label,
                 variant, baseline_label, idiom, label_type):
        self.word = word
        self.vector = vector
        self.variant = variant
        self.predictions = defaultdict(list)

        if translation_label == "paraphrase":
            self.translation_label = 1
        else:
            self.translation_label = 0

        if magpie_label == "figurative":
            self.magpie_label = 1
        else:
            self.magpie_label = 0

        if baseline_label == "figurative":
            self.baseline_label = 1
        else:
            self.baseline_label = 0

        if label_type == "translation":
            self.label = self.translation_label
        else:
            self.label = self.magpie_label


class Probe(torch.nn.Module):
    def __init__(self, in_features, l0=False):
        super().__init__()
        self.l0 = l0
        self.dropout = torch.nn.Dropout(0.1)
        if l0:
            self.weights1 = HardConcreteLinear(in_features, 1)
        else:
            self.weights1 = torch.nn.Linear(in_features, 1)

    def forward(self, input_vec):
        input_vec = self.dropout(input_vec)
        output = self.weights1(input_vec)
        return torch.sigmoid(output).squeeze(-1)


def extract_encodings(indices, label_type, key, keywords_only=True):
    
    assert label_type in ["magpie", "translation"], "label_type is unknown"
    assert key in ["encodings_idiom", "encodings_context"]
    classifier = Classifier(
        "../data/magpie-corpus/idiom_annotations_automated_spacy_translated.tsv")
    samples = []
    for i in indices:
        if not os.path.exists(f"per_idiom/{i}_pred.pickle"):
            continue

        encodings = pickle.load(open(f"per_idiom/{i}_pred.pickle", 'rb'))
        sample_info = open(f"per_idiom/{i}.tsv", encoding="utf-8").readlines()
        if not sample_info:
            continue

        prds = open(f"per_idiom/{i}_pred.txt", encoding="utf-8").readlines()
        prds = {x.split('\t')[0] : y.strip() for x, y in zip(sample_info, prds)} 
        idiom = sample_info[0].split("\t")[2]

        if not classifier.contains(idiom):
            continue

        lemmatized_keywords = []
        for w2 in nlp(" ".join(classifier.en_keywords[idiom])):
            lemmatized_keywords.append(w2.lemma_)

        sample_info = {x.split("\t")[0] : tuple(x.split("\t")[1:])
                       for x in sample_info}
        labels = [x[-2] for x in sample_info.values()]
        random.shuffle(labels)
        common_labels = Counter(labels).most_common(1)[0][0]

        if key not in encodings:
            continue
        for sent in encodings[key]:
            if sent not in prds:
                continue
            for word in encodings[key][sent]:
                if key == "encodings_idiom" and keywords_only and word[1:].lower() not in lemmatized_keywords:
                    continue
                vector = torch.FloatTensor(encodings[key][sent][word])
                translation_label = classifier(idiom, sent, prds[sent])
                samples.append(Sample(
                    word, vector, translation_label,
                    sample_info[sent][-2],
                    sample_info[sent][-1],
                    common_labels, idiom, label_type))
    return samples
