import sys
sys.path.append('../data/')
sys.path.append('../amnesic_probing/')
import logging
import numpy as np
import random
import seaborn as sns
import cca_core
from matplotlib import pyplot as plt
from data import extract_sentences
from classifier import Classifier
from collections import defaultdict, Counter
from transformers import MarianTokenizer


def visualise(all_coefs, filename, caption):
    """
    Visualise how  hidden representations change over layers.
    Args:
        - all_coefs (dict): keys as vocab setup, coefs as values
        - filename (str): for storing the figure
    """
    plt.figure(figsize=(4.5, 4.5))
    print(all_coefs)
    for category, label in vocab_setups:
        colours = sns.color_palette("viridis", 8)
        sns.scatterplot(x=[1, 2, 3, 4, 5], y=all_coefs[category],
                        color=colours[category], label=label)
        sns.lineplot(x=[1, 2, 3, 4, 5], y=all_coefs[category],
                     color=colours[category], alpha=0.5)
    plt.xlabel("layers")
    plt.ylabel(caption)
    plt.xticks([1, 2, 3, 4, 5], ["1-2", "2-3", "3-4", "4-5", "5-6"])
    plt.legend(bbox_to_anchor=(0.95, 1.05), frameon=False)
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()


def two_step_svcca(freq_subsets, samples_for_estimation):
    """
    Perform CCA by projecting hidden states with pre-computed CCA data.
    Args:
        freq_subsets: dict(layer: dict(category: vectors))
        samples_for_estimation: list of Sample objects, for the items
            we'll use for estimation of CCA projection matrices.
    Saves a visualisation of the cosine similarities obtained to file.
    """
    cca_results = []

    # Collect any hidden states for estimating the SVCCA transformation
    random.shuffle(samples_for_estimation)
    for i in range(1, 7):
        logging.info(f"Collecting CCA coefs for estimation for layer {i}.")
        layer_a = []
        layer_b = []
        for sample in samples_for_estimation:
            words = sample.tokenised_sentence.split()
            for word, hid in zip(words, sample.hidden_states[i]):
                layer_a.append(hid.tolist())
            for word, hid in zip(words, sample.hidden_states[i - 1]):
                layer_b.append(hid.tolist())

        layer_a = np.array(layer_a).transpose(1, 0)[:, :100000]
        layer_b = np.array(layer_b).transpose(1, 0)[:, :100000]
        logging.info(layer_a.shape)
        cca = cca_core.get_cca_similarity(
            layer_a, layer_b,
            verbose=False, epsilon=1e-7, threshold=1)
        cca_results.append(cca)

    all_coefs = defaultdict(list)
    for category in [0, 1, 2, 3, 4]:
        coefs = []
        for i in range(2, 7):
            layer_a = np.array(freq_subsets[i][category]).transpose(1, 0)
            layer_b = np.array(freq_subsets[i - 1][category]).transpose(1, 0)
            coefs.append(cca_core.compute_cosine_sim(
                layer_a[:, :5120], layer_b[:, :5120], cca_results[i - 1]))
        all_coefs[category] = coefs

    visualise(all_coefs, "two_step_cca", "Two-step CCA")


def svcca(freq_subsets):
    """
    Perform conventional CCA using Google's CCA core functionality.
    Args:
        freq_subsets: dict(layer: dict(category: vectors))
    Saves a visualisation of the CCA coefficients obtained to file.
    """
    all_coefs = defaultdict(list)
    for category in [0, 1, 2, 3, 4]:
        coefs = []
        for i in range(2, 7):
            layer_a = np.array(freq_subsets[i][category]).transpose(1, 0)
            layer_b = np.array(freq_subsets[i - 1][category]).transpose(1, 0)
            logging.info(layer_a.shape)
            logging.info(layer_b.shape)
            # Set threshold to 1, because we also will not remove coefficients
            # in the two-step version.
            cca = cca_core.get_cca_similarity(
                layer_a[:, :5120], layer_b[:, :5120],
                verbose=False, epsilon=1e-7, threshold=1)
            coefs.append(cca["mean"][0])
        all_coefs[category] = coefs

    visualise(all_coefs, "cca", "CCA")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    classifier = Classifier(
        "../data/keywords/idiom_keywords_translated_nl.tsv")
    mname = "Helsinki-NLP/opus-mt-en-nl"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    sns.set_context("talk")

    # Load all hidden representations of idioms
    samples_for_analysis = extract_sentences(
        range(0, 500), classifier, tokenizer, use_tqdm=True,
        store_hidden_states=True, data_folder="../data/magpie/nl")
    samples_for_estimation = extract_sentences(
        range(500, 1000), classifier, tokenizer, use_tqdm=True,
        store_hidden_states=True, data_folder="../data/magpie/nl")

    # Shuffle the examples
    random.seed(1)
    random.shuffle(samples_for_estimation)
    random.shuffle(samples_for_analysis)

    # Collect words that are relatively low frequent
    vocab = []
    for s in samples_for_analysis:
        vocab.extend(s.tokenised_sentence.split())
    vocab = Counter(vocab)
    vocab = list(set(x for x in vocab if vocab[x] >= 8 and vocab[x] <= 128))
    logging.info(f"Found {len(vocab)} words with frequency between 16 - 100.")

    # We want five sets of vocabularies that have approx. the same amoun of vecs
    vocab_setups = [(0, "40 x 128"), (1, "80 x 64"), (2, "160 x 32"),
                    (3, "320 x 16"), (4, "640 x 8")]
    vocab_size = {0: (40, 128), 1: (80, 64), 2: (160, 32),
                  3: (320, 16), 4: (640, 8)}
    vocab_sets = dict()
    random.shuffle(vocab)
    for cat in [0, 1, 2, 3, 4]:
        vocab_sets[cat] = vocab[:vocab_size[cat][0]]

    # Now collect vectors for each of these categories according to the right
    # frequencies
    freq_subsets = defaultdict(lambda: defaultdict(list))
    vocab_freqs = defaultdict(lambda: defaultdict(Counter))
    for cat in [0, 1, 2, 3, 4]:
        max_freq = vocab_size[cat][1]
        for layer in range(7):
            for s in samples_for_analysis:
                snt = s.tokenised_sentence.split()
                hidden_states = s.hidden_states[layer]
                for word, hidden_state in zip(snt, hidden_states):
                    frequency = vocab_freqs[layer][cat][word]
                    if word in vocab_sets[cat] and frequency < max_freq:
                        freq_subsets[layer][cat].append(hidden_state.tolist())
                        vocab_freqs[layer][cat][word] += 1

    two_step_svcca(freq_subsets, samples_for_estimation)
    svcca(freq_subsets)
