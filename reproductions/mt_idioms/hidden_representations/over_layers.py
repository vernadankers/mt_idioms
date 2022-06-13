import sys
sys.path.append('../data/')
sys.path.append('../amnesic_probing/')
from probing import set_seed
import random
from collections import defaultdict
import torch
import cca_core
import numpy as np
import logging
import pickle
import argparse
from transformers import MarianTokenizer
from classifier import Classifier
from data import extract_sentences


def precompute_cca(classifier, tokenizer, language):
    # Load all hidden representations of idioms
    logging.info("Collecting data for estimating CCA, this takes a while...")
    samples_for_cca = extract_sentences(
        range(0, 1727), classifier, tokenizer, use_tqdm=False,
        data_folder=f"../data/magpie/{language}",
        store_hidden_states=True, get_verb_idioms=True)
    logging.info(f"CCA samples: {len(samples_for_cca)}")

    logging.info("Estimating CCA directions, this takes a while...")
    # Perform CCA on a separate set of samples
    cca_results = []
    random.shuffle(samples_for_cca)
    for i in range(1, 7):
        layer1 = []
        layer2 = []
        for s in samples_for_cca:
            hidden_states = s.hidden_states[i - 1].tolist()
            layer1.extend([h for h in hidden_states if sum(h) != 0])
            hidden_states = s.hidden_states[i].tolist()
            layer2.extend([h for h in hidden_states if sum(h) != 0])
        layer1 = np.array(layer1).transpose(1, 0)
        layer2 = np.array(layer2).transpose(1, 0)
        cca = cca_core.get_cca_similarity(
            layer1[:, :100000], layer2[:, :100000], epsilon=1e-7, verbose=False)
        cca_results.append(cca)
    return cca_results


def main(language):
    # Step 1: load the data
    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)

    cca_results = precompute_cca(classifier, tokenizer, language)

    logging.info("Collecting data for words to analyse, this takes a while...")
    samples_to_analyse = extract_sentences(
        range(0, 1727), classifier, tokenizer, use_tqdm=False,
        data_folder=f"../data/magpie/{language}",
        store_hidden_states=True)
    logging.info(f"Samples to analyse: {len(samples_to_analyse)}")

    logging.info("Extracting hidden states to analyse, this takes a while...")
    # Now analyse the remaining samples
    all_coefs = defaultdict(list)
    hidden_states = defaultdict(lambda: defaultdict(list))
    for layer_ in range(7):
        for s in samples_to_analyse:
            # Collect PIE nouns
            for i in s.index_select(1, tags=["NOUN"]):
                h = s.hidden_states[layer_][i].tolist()
                hidden_states[layer_][(
                    1, s.magpie_label, s.translation_label)].append(h)
                hidden_states[layer_][(1, s.magpie_label)].append(h)
                hidden_states[layer_][(1, s.translation_label)].append(h)

            # And non-PIE nouns, only use those for which the hidden state is
            # not zeroed out. The zeroing was to reduce data size for storage
            non_pie_nouns = [z for z in s.index_select(0, tags=["NOUN"])
                             if torch.sum(s.hidden_states[layer_][z]) != 0]
            for i in non_pie_nouns:
                h = s.hidden_states[layer_][i].tolist()
                hidden_states[layer_][(
                    0, s.magpie_label, s.translation_label)].append(h)
                hidden_states[layer_][(0, s.magpie_label)].append(h)
                hidden_states[layer_][(0, s.translation_label)].append(h)

    logging.info("Computing similarity between layers with precomputed CCA.")
    for category in hidden_states[0].keys():
        coefs = []
        for i in range(1, 7):
            layer1 = np.array(
                hidden_states[i - 1][category]).transpose(1, 0)
            layer2 = np.array(
                hidden_states[i][category]).transpose(1, 0)
            coefs.append(cca_core.compute_cosine_sim(
                layer1, layer2, cca_results[i - 1]))
        all_coefs[category] = coefs

    pickle.dump(dict(all_coefs), open(
        f"data/{language}/over_layers.pickle", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="nl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(1)
    main(args.language)
