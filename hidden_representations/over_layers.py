from transformers import MarianTokenizer
import argparse
import pickle
import logging
import numpy as np
import cca_core
import tqdm
import torch
from collections import defaultdict, Counter
import random
import os
import sys
sys.path.append('../data/')
from classifier import Classifier
from data import extract_sentences


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


logging.basicConfig(level=logging.INFO)
set_seed(1)


def compute_cosine_sim(data1, data2, cca_results):
    cacts1 = data1 - cca_results["neuron_means1"]
    cacts2 = data2 - cca_results["neuron_means2"]

    P = cca_results["full_coef_x"]
    data1 = np.dot(
        np.dot(P.T, np.dot(P, cca_results["full_invsqrt_xx"])), cacts1)
    P = cca_results["full_coef_y"]
    data2 = np.dot(
        np.dot(P.T, np.dot(P, cca_results["full_invsqrt_yy"])), cacts2)
    cos = torch.nn.CosineSimilarity(dim=0)
    cos = cos(torch.FloatTensor(data1), torch.FloatTensor(data2))
    logging.info(cos.shape)
    return torch.mean(cos).item()


def main(language):
    # Step 1: load the data
    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)

    # Load all hidden representations of idioms
    samples_for_cca = extract_sentences(
        range(0, 1727), classifier, tokenizer, use_tqdm=True, data_folder=f"../data/magpie/{language}",
        store_hidden_states=True, get_verb_idioms=True)
    sents1 = [s.sentence for s in samples_for_cca]

    samples_to_analyse = extract_sentences(
        range(0, 1727), classifier, tokenizer, use_tqdm=True, data_folder=f"../data/magpie/{language}",
        store_hidden_states=True)
    sents2 = [s.sentence for s in samples_to_analyse]
    logging.info(f"CCA samples: {len(samples_for_cca)}, Samples to analyse: {len(samples_to_analyse)}, Intersection: {len(set(sents1).intersection(set(sents2)))}")

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
                hidden_states[layer_][(
                    1, s.magpie_label, s.translation_label)].append(h)
                hidden_states[layer_][(1, s.magpie_label)].append(h)
                hidden_states[layer_][(1, s.translation_label)].append(h)

            # And non-PIE nouns
            for i in [z for z in s.index_select(0, tags=["NOUN"]) if torch.sum(s.hidden_states[layer_][z]) != 0]:
                h = s.hidden_states[layer_][i]
                hidden_states[layer_][(
                    0, s.magpie_label, s.translation_label)].append(h)
                hidden_states[layer_][(0, s.magpie_label)].append(h)
                hidden_states[layer_][(0, s.translation_label)].append(h)

    for category in hidden_states[0].keys():
        coefs = []
        for i in range(1, 7):
            indices = list(range(len(hidden_states[i][category])))
            random.shuffle(indices)
            indices = indices[:100000]
            layer1 = torch.stack(
                hidden_states[i - 1][category], dim=0)[indices].transpose(0, 1).numpy()
            layer2 = torch.stack(hidden_states[i][category], dim=0)[
                                 indices].transpose(0, 1).numpy()
            coefs.append(compute_cosine_sim(
                layer1, layer2, cca_results[i - 1]))
        all_coefs[category] = coefs

    pickle.dump(dict(all_coefs), open(f"data/{language}/over_layers.pickle", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="nl")
    args = parser.parse_args()
    main(args.language)
