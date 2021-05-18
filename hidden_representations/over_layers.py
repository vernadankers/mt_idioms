import sys
sys.path.append('../data/')
sys.path.append('../amnesic_probing/')
from probing import set_seed
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
set_seed(1)


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


def main():
    # Load all hidden representations of idioms
    samples_for_cca = extract_sentences(
        range(0, 1727), use_tqdm=True, data_folder="../data/magpie",
        store_hidden_states=True, get_verb_idioms=True)
    logging.info(len(samples_for_cca))
    sents1 = [s.sentence for s in samples_for_cca]

    samples_to_analyse = extract_sentences(
        range(0, 1727), use_tqdm=True, data_folder="../data/magpie",
        store_hidden_states=True)
    logging.info(len(samples_to_analyse))
    sents2 = [s.sentence for s in samples_to_analyse]
    logging.info(len(set(sents1).intersection(set(sents2))))

    # Perform CCA on a separate set of samples
    cca_results = []
    random.shuffle(samples_for_cca)
    for i in range(1, 7):
        layer1 = []
        layer2 = []
        for s in samples_for_cca:
            layer1.extend(s.hidden_states[i - 1].tolist())
            layer2.extend(s.hidden_states[i].tolist())
        layer1 = torch.FloatTensor(layer1).transpose(0, 1).numpy()
        layer2 = torch.FloatTensor(layer2).transpose(0, 1).numpy()
        cca = cca_core.get_cca_similarity(
            layer1[:, :100000], layer2[:, :100000], epsilon=1e-6)
        cca_results.append(cca)

    # Now analyse the remaining samples
    all_coefs = defaultdict(list)
    hidden_states = defaultdict(lambda: defaultdict(list))
    random.seed(1)
    random.shuffle(samples_to_analyse)
    for layer_ in range(7):
        for s in samples_to_analyse:
            # Collect PIE nouns
            for i in s.index_select(1, tags=["NOUN"]):
                h = s.hidden_states[layer_][i]
                hidden_states[layer_][(1, s.magpie_label, s.translation_label)].append(h)
                hidden_states[layer_][(1, s.magpie_label)].append(h)
                hidden_states[layer_][(1, s.translation_label)].append(h)

            # And non-PIE nouns
            for i in s.index_select(0, tags=["NOUN"]):
                h = s.hidden_states[layer_][i]
                hidden_states[layer_][(0, s.magpie_label, s.translation_label)].append(h)
                hidden_states[layer_][(0, s.magpie_label)].append(h)
                hidden_states[layer_][(0, s.translation_label)].append(h)

    for category in hidden_states[0].keys():
        coefs = []
        for i in range(1, 7):
            indices = list(range(len(hidden_states[i][category])))
            random.shuffle(indices)
            indices = indices[:100000]
            layer1 = torch.stack(hidden_states[i - 1][category], dim=0)[indices].transpose(0, 1).numpy()
            layer2 = torch.stack(hidden_states[i][category], dim=0)[indices].transpose(0, 1).numpy()
            coefs.append(compute_cosine_sim(layer1, layer2, cca_results[i - 1]))
        all_coefs[category] = coefs

    pickle.dump(dict(all_coefs), open("data/over_layers.pickle", "wb"))


if __name__ == "__main__":
    main()
