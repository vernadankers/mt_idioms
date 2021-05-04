import sys
import argparse
import random
from collections import defaultdict
import logging
import pickle
import numpy as np
from tqdm import tqdm
import torch
from cca_core import get_cca_similarity
sys.path.append('../data/')
from data import extract_sentences



def get_cca_results(layer, samples, samples_masked, label, neighbourhood):
    data1 = []
    data2 = []

    indices = list(range(len(samples_masked)))
    random.shuffle(indices)
    samples = [samples[i] for i in indices]
    samples_masked = [samples_masked[i] for i in indices]

    for s, s_masked in zip(samples, samples_masked):
        indices = s_masked.index_select(0, neighbours_only=True,
            neighbourhood=neighbourhood, context_context=True, layer=layer)
        if not indices:
            continue
        if not torch.any(s.hidden_states[1:] != s_masked.hidden_states[1:]):
            logging.info(s.idiom)
            continue
        n_tokens = len(s.tokenised_annotation)
        vecs = torch.FloatTensor(s.hidden_states[layer, :n_tokens])
        vecs = vecs.index_select(dim=0, index=torch.LongTensor(indices))
        vecs_masked = torch.FloatTensor(s_masked.hidden_states[layer, :n_tokens])
        vecs_masked = vecs_masked.index_select(dim=0, index=torch.LongTensor(indices))
        data1.extend(vecs)
        data2.extend(vecs_masked)

    indices = list(range(len(data1)))
    random.shuffle(indices)
    data1 = torch.stack(data1)[indices[:100000]].transpose(0, 1).numpy()
    data2 = torch.stack(data2)[indices[:100000]].transpose(0, 1).numpy()
    return get_cca_similarity(data1, data2, epsilon=1e-6)


def collect_data_per_layer(layer, samples, samples_masked, label,
                           neighbours_only, neighbourhood, context_context):
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
            neighbourhood=neighbourhood, context_context=context_context, layer=layer)

        if not indices or not torch.any(s.hidden_states[1:] != s_masked.hidden_states[1:]):
            continue
        n_tokens = len(s.tokenised_annotation)
        indices = torch.LongTensor(indices)

        # Get regular hidden representations
        vecs = torch.FloatTensor(s.hidden_states[layer, :n_tokens])
        vecs = vecs.index_select(dim=0, index=indices)

        vecs_masked = torch.FloatTensor(s_masked.hidden_states[layer, :n_tokens])
        vecs_masked = vecs_masked.index_select(dim=0, index=indices)

        # Collect hidden states for pairs of MAGPIE and translation labels
        label_pair = (s.magpie_label, s.translation_label)
        idioms_contained[label_pair].append(s.idiom)
        data1[label_pair].extend(vecs)
        data2[label_pair].extend(vecs_masked)

        # Collect hidden states per MAGPIE label
        idioms_contained[s.magpie_label].append(s.idiom)
        data1[s.magpie_label].extend(vecs)
        data2[s_masked.magpie_label].extend(vecs_masked)

        # Collect hidden states per translation label
        idioms_contained[s.translation_label].append(s.idiom)
        data1[s.translation_label].extend(vecs)
        data2[s_masked.translation_label].extend(vecs_masked)
    return data1, data2


def compute_cosine_sim(data1, data2, cca_results):
    cacts1 = data1 - np.mean(data1, axis=1, keepdims=True)
    cacts2 = data2 - np.mean(data2, axis=1, keepdims=True)

    P = cca_results["full_coef_x"]
    data1 = np.dot(np.dot(P.T, np.dot(P, cca_results["full_invsqrt_xx"])), cacts1)
    P = cca_results["full_coef_y"]
    data2 = np.dot(np.dot(P.T, np.dot(P, cca_results["full_invsqrt_yy"])), cacts2)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos = cos(torch.FloatTensor(data1), torch.FloatTensor(data2))
    return torch.mean(cos).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighbours_only", action="store_true")
    parser.add_argument("--neighbourhood", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--m", type=int, default=1728)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--context_context", action="store_true")
    parser.add_argument("--masked_folder", type=str,
                        default="../data/magpie/masked_idiom")
    parser.add_argument("--save_file", type=str, default="svcca.pickle")
    args = parser.parse_args()

    # Load sentences from pickled files with hidden representations
    logging.basicConfig(level="INFO")
    samples, samples_masked = [], []
    for i in tqdm(list(range(args.n, args.m, args.k))):
        without_mask = extract_sentences(
            [i], use_tqdm=False, data_folder="../data/magpie/masked_regular",
            influence_setup=True)
        with_mask = extract_sentences(
            [i], use_tqdm=False, data_folder=args.masked_folder,
            influence_setup=True)
        if len(without_mask) == len(with_mask):
            samples.extend(without_mask)
            samples_masked.extend(with_mask)

    # Load the sentences with verb idioms to measure the CCA results on
    samples_cca, samples_cca_masked = [], []
    for i in tqdm(list(range(args.n, args.m, args.k))):
        without_mask = extract_sentences(
            [i], use_tqdm=False, data_folder="../data/magpie/masked_regular",
            influence_setup=True, get_verb_idioms=True)
        with_mask = extract_sentences(
            [i], use_tqdm=False, data_folder="../data/magpie/masked_context",
            influence_setup=True, get_verb_idioms=True)

        if len(without_mask) == len(without_mask):
            samples_cca.extend(without_mask)
            samples_cca_masked.extend(with_mask)

    svcca_similarities = defaultdict(list)
    for layer in range(0, 6):
        logging.info(f"Performing SVCCA for layer {layer}.")
        vectors, vectors_masked = collect_data_per_layer(
            layer, samples, samples_masked, args.label,
            args.neighbours_only, args.neighbourhood, args.context_context)

        cca_results = get_cca_results(
            layer, samples_cca, samples_cca_masked, args.label, args.neighbourhood)

        # Per subset label, compute similarities
        for subset_name in vectors:
            vectors[subset_name] = np.stack(vectors[subset_name], axis=1)
            vectors_masked[subset_name] = np.stack(vectors_masked[subset_name], axis=1)

            # Report how many samples are in the different subsets
            if layer == 0:
                logging.info(f"{subset_name}, {vectors[subset_name].shape}")

            # Two-step SVCCA
            two_step_svcca = compute_cosine_sim(
                vectors[subset_name], vectors_masked[subset_name], cca_results)
            svcca_similarities[subset_name].append(two_step_svcca)

    pickle.dump(svcca_similarities, open(args.save_file, 'wb'))
