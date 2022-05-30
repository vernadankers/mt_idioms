import sys
sys.path.append('../data/')
from cca_core import get_cca_similarity, compute_cosine_sim
import torch
from tqdm import tqdm
import numpy as np
import pickle
import logging
from collections import defaultdict
import random
import argparse
from classifier import Classifier
from transformers import MarianTokenizer
from data import extract_sentences


def get_cca_results(layer, samples, samples_masked, neighbourhood):
    """
    Get the CCA projection matrices for a given layer, using samples before
    and after masking.
    Args:
        layer (int)
        samples (list of Sentence objects)
        samples_masked (list of Sentence objects)
        neighbourhood (int): neighbourhood size
    Returns:
        CCA results dictionary (see cca_core.py for details)
    """
    data1 = []
    data2 = []

    indices = list(range(len(samples_masked)))
    random.shuffle(indices)
    samples = [samples[i] for i in indices]
    samples_masked = [samples_masked[i] for i in indices]

    for s, s_masked in zip(samples, samples_masked):
        assert s.sentence == s_masked.sentence
        indices = s_masked.index_select(
            0, neighbours_only=True, neighbourhood=neighbourhood,
            context_of_mask=True, layer=layer)

        if not indices:
            continue
        if not torch.any(s.hidden_states[1:] != s_masked.hidden_states[1:]):
            logging.error(
                f"For {s.idiom}: states didn't change after masking.")

        n_tokens = len(s.tokenised_annotation)
        vecs = torch.FloatTensor(s.hidden_states[layer, :n_tokens])
        vecs_masked = torch.FloatTensor(
            s_masked.hidden_states[layer, :n_tokens])

        # When gathering data we zeroed out some vectors reduce storage space
        # Now we do not want to use those vectors here.
        indices = [x for x in indices if torch.sum(
            vecs[x]) != 0 and torch.sum(vecs_masked[x]) != 0]

        vecs = vecs.index_select(dim=0, index=torch.LongTensor(indices))
        vecs_masked = vecs_masked.index_select(
            dim=0, index=torch.LongTensor(indices))
        data1.extend(vecs)
        data2.extend(vecs_masked)

    indices = list(range(len(data1)))
    random.shuffle(indices)
    logging.info(f"Dataset size: {len(indices)}")
    data1 = torch.stack(data1)[indices[:100000]].transpose(0, 1).numpy()
    data2 = torch.stack(data2)[indices[:100000]].transpose(0, 1).numpy()
    logging.info(f"Performing CCA using {data1.shape} dimensions")
    return get_cca_similarity(data1, data2, epsilon=1e-7, verbose=False)


def collect_data_per_layer(layer, samples, samples_masked, label,
                           neighbours_only, neighbourhood):
    """
    """
    data1 = defaultdict(list)
    data2 = defaultdict(list)
    idioms_contained = defaultdict(list)

    for s, s_masked in zip(samples, samples_masked):
        assert s.sentence == s_masked.sentence
        if s.translation_label in ["none", "copied"]:
            continue

        # Select the tokens of interest
        indices = s_masked.index_select(
            label, neighbours_only=neighbours_only,
            neighbourhood=neighbourhood, context_of_mask=True, layer=layer)

        if not indices or not torch.any(s.hidden_states[1:] != s_masked.hidden_states[1:]):
            continue
        n_tokens = len(s.tokenised_annotation)

        # Get regular hidden representations
        vecs = torch.FloatTensor(s.hidden_states[layer, :n_tokens])
        vecs_masked = torch.FloatTensor(
            s_masked.hidden_states[layer, :n_tokens])

        indices = [x for x in indices if torch.sum(
            vecs[x]) != 0 and torch.sum(vecs_masked[x]) != 0]
        vecs = vecs.index_select(dim=0, index=torch.LongTensor(indices))
        vecs_masked = vecs_masked.index_select(
            dim=0, index=torch.LongTensor(indices))

        # Collect hidden states for pairs of magpie_de and translation labels
        label_pair = (s.magpie_label, s.translation_label)
        idioms_contained[label_pair].append(s.idiom)
        data1[label_pair].extend(vecs)
        data2[label_pair].extend(vecs_masked)

        # Collect hidden states per magpie_de label
        idioms_contained[s.magpie_label].append(s.idiom)
        data1[s.magpie_label].extend(vecs)
        data2[s_masked.magpie_label].extend(vecs_masked)

        # Collect hidden states per translation label
        idioms_contained[s.translation_label].append(s.idiom)
        data1[s.translation_label].extend(vecs)
        data2[s_masked.translation_label].extend(vecs_masked)
    return data1, data2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighbours_only", action="store_true")
    parser.add_argument("--neighbourhood", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=1727)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--masked_folder", type=str,
                        default="../data/magpie_de/masked_idiom")
    parser.add_argument("--save_file", type=str, default="cca.pickle")
    parser.add_argument("--language", type=str, default="nl")
    args = parser.parse_args()

    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{args.language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{args.language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)

    # Load sentences from pickled files with hidden representations
    logging.basicConfig(level="INFO")
    samples, samples_masked = [], []
    for i in tqdm(list(range(args.start, args.stop, args.step))):
        without_mask = extract_sentences(
            [i], classifier, tokenizer, use_tqdm=False,
            data_folder=f"../data/magpie/{args.language}",
            data_folder2=f"../data/magpie/{args.language}/mask_regular",
            influence_setup=True, stacked_by_layer=True)
        with_mask = extract_sentences(
            [i], classifier, tokenizer, use_tqdm=False,
            data_folder=f"../data/magpie/{args.language}",
            data_folder2=args.masked_folder,
            influence_setup=True, stacked_by_layer=True)

        assert len(without_mask) == len(with_mask)
        samples.extend(without_mask)
        samples_masked.extend(with_mask)

    # Load the sentences with verb idioms to measure the CCA results on
    samples_cca, samples_cca_masked = [], []
    for i in tqdm(list(range(args.start, args.stop, args.step))):
        without_mask = extract_sentences(
            [i], classifier, tokenizer, use_tqdm=False,
            data_folder=f"../data/magpie/{args.language}",
            data_folder2=f"../data/magpie/{args.language}/mask_regular",
            influence_setup=True, get_verb_idioms=True, stacked_by_layer=True)
        with_mask = extract_sentences(
            [i], classifier, tokenizer, use_tqdm=False,
            data_folder=f"../data/magpie/{args.language}",
            data_folder2=f"../data/magpie/{args.language}/mask_context",
            influence_setup=True, get_verb_idioms=True, stacked_by_layer=True)

        assert len(without_mask) == len(without_mask)
        samples_cca.extend(without_mask)
        samples_cca_masked.extend(with_mask)

    logging.info(len(samples_cca), len(samples_masked))

    cca_similarities = defaultdict(list)
    for layer in range(0, 6):
        logging.info(f"Performing CCA for layer {layer}.")
        vectors, vectors_masked = collect_data_per_layer(
            layer, samples, samples_masked, args.label,
            args.neighbours_only, args.neighbourhood)

        cca_results = get_cca_results(
            layer, samples_cca, samples_cca_masked, args.neighbourhood)

        # Per subset label, compute similarities
        for subset_name in vectors:
            vectors[subset_name] = np.stack(vectors[subset_name], axis=1)
            vectors_masked[subset_name] = np.stack(
                vectors_masked[subset_name], axis=1)

            # Report how many samples are in the different subsets
            if layer == 0:
                logging.info(f"{subset_name}, {vectors[subset_name].shape}")

            # Two-step CCA
            two_step_cca = compute_cosine_sim(
                vectors[subset_name], vectors_masked[subset_name], cca_results)
            cca_similarities[subset_name].append(two_step_cca)

    pickle.dump(cca_similarities, open(args.save_file, 'wb'))
