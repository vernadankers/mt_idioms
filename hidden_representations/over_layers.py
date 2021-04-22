import sys
sys.path.append('../data/')
sys.path.append('../probing/')
import os
import random
from collections import defaultdict, Counter
import torch
import tqdm
from classifier import Classifier
import cca_core
from data import extract_sentences
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.INFO)



def compute_cosine_sim(data1, data2, cca_results):
    """Use CCA to apply to a new set of samples."""
    k = 512
    cacts1 = data1 - np.mean(data1, axis=1, keepdims=True)
    cacts2 = data2 - np.mean(data2, axis=1, keepdims=True)

    data1 = np.dot(
        np.dot(cca_results["full_coef_x"][:k].T,
               np.dot(cca_results["full_coef_x"][:k],
                      cca_results["full_invsqrt_xx"])),
        cacts1)
    data2 = np.dot(
        np.dot(cca_results["full_coef_y"][:k].T,
               np.dot(cca_results["full_coef_y"][:k],
                      cca_results["full_invsqrt_yy"])),
        cacts2)
    data1 = data1 - np.mean(data1, axis=1, keepdims=True)
    data2 = data2 - np.mean(data2, axis=1, keepdims=True)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos = cos(torch.FloatTensor(data1), torch.FloatTensor(data2))
    return torch.mean(cos).item()


# Load all hidden representations of idioms
samples_for_cca = extract_sentences(
    range(0, 50), use_tqdm=True, data_folder="../data/magpie",
    store_hidden_states=True, get_verb_idioms=True)
logging.info(len(samples_for_cca))
sents1 = [s.sentence for s in samples_for_cca]

samples_to_analyse = extract_sentences(
    range(0, 50), use_tqdm=True, data_folder="../data/magpie",
    store_hidden_states=True)
logging.info(len(samples_to_analyse))
sents2 = [s.sentence for s in samples_to_analyse]
logging.info(len(set(sents1).intersection(set(sents2))))

# Perform CCA on a separate set of samples

per_setup_labels = defaultdict(list)

cca_results = []
random.shuffle(samples_for_cca)
for i in range(1, 7):
    layer1 = []
    layer2 = []
    for s in samples_for_cca:
        layer1.extend(s.hidden_states[i-1].tolist())
        layer2.extend(s.hidden_states[i].tolist())
    layer1 = torch.FloatTensor(layer1).transpose(0, 1).numpy()
    layer2 = torch.FloatTensor(layer2).transpose(0, 1).numpy()
    cca = cca_core.get_cca_similarity(
        layer1[:, :50000], layer2[:, :50000], epsilon=1e-6)
    cca_results.append(cca)

for TAGS in [["NOUN"]]:
    # Now analyse the remaining samples
    all_coefs = defaultdict(list)
    freq_subsets = defaultdict(lambda: defaultdict(list))
    random.seed(1)
    random.shuffle(samples_to_analyse)
    for layer_ in range(7):
        for s in samples_to_analyse:
            for word, hidden_state, a, tag in zip(
                s.tokenised_sentence.split(), s.hidden_states[layer_],
                s.tokenised_annotation, s.pos_tags):
                if a == 0 or (tag not in TAGS if TAGS else False):
                    continue
                freq_subsets[layer_][(s.magpie_label, s.translation_label)].append(hidden_state)
                freq_subsets[layer_][s.magpie_label].append(hidden_state)
                freq_subsets[layer_][s.translation_label].append(hidden_state)

    for category in freq_subsets[0].keys():
        coefs = []
        for i in range(1, 7):
            indices = list(range(len(freq_subsets[i][category])))
            random.shuffle(indices)
            indices = indices[:50000]
            layer1 = torch.stack(freq_subsets[i-1][category], dim=0)[indices].transpose(0, 1).numpy()
            layer2 = torch.stack(freq_subsets[i][category], dim=0)[indices].transpose(0, 1).numpy()
            coefs.append(compute_cosine_sim(layer1, layer2, cca_results[i - 1]))
        all_coefs[category] = coefs

    per_setup_labels[(1, "_".join(TAGS) if TAGS else "all")] = dict(all_coefs)

for TAGS in [["NOUN"]]:
    # Now analyse the remaining samples
    all_coefs = defaultdict(list)
    freq_subsets = defaultdict(lambda: defaultdict(list))
    random.seed(1)
    random.shuffle(samples_to_analyse)
    for layer_ in range(7):
        for s in samples_to_analyse:
            for word, hidden_state, a, tag in zip(
                s.tokenised_sentence.split(), s.hidden_states[layer_],
                s.tokenised_annotation, s.pos_tags):
                if a == 1 or (tag not in TAGS if TAGS else False):
                    continue
                freq_subsets[layer_][(s.magpie_label, s.translation_label)].append(hidden_state)
                freq_subsets[layer_][s.magpie_label].append(hidden_state)
                freq_subsets[layer_][s.translation_label].append(hidden_state)

    for category in freq_subsets[0].keys():
        coefs = []
        for i in range(1, 7):
            indices = list(range(len(freq_subsets[i][category])))
            random.shuffle(indices)
            indices = indices[:50000]
            layer1 = torch.stack(freq_subsets[i-1][category], dim=0)[indices].transpose(0, 1).numpy()
            layer2 = torch.stack(freq_subsets[i][category], dim=0)[indices].transpose(0, 1).numpy()
            coefs.append(compute_cosine_sim(layer1, layer2, cca_results[i - 1]))
        all_coefs[category] = coefs

    per_setup_labels[(0, "_".join(TAGS) if TAGS else "all")] = dict(all_coefs)

pickle.dump(per_setup_labels, open("data/over_layers.pickle", "wb"))
