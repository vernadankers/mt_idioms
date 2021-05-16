import sys
from collections import defaultdict
import random
import argparse
import logging
import pickle
import torch
from torch import LongTensor as LT
sys.path.append('../data/')
from data import extract_sentences
random.seed(1)


def main(mode, start, stop, step):
    # Step 1: load the data
    sentences = extract_sentences(
        range(start, stop, step), use_tqdm=False,
        data_folder="../data/magpie", store_attention=True)

    # Restrict the data used to...
    # - identical are matches labelled as identical by MAGPIE
    # - intersection are PIEs that are both in fig-par and lit-wfw
    if mode == "identical":
        sentences = [x for x in sentences if x.variant == "identical"]
    elif mode == "intersection":
        lit_idioms = {x.idiom for x in sentences if x.translation_label ==
                      "word-by-word" and x.magpie_label == "literal"}
        fig_idioms = {x.idiom for x in sentences if x.translation_label ==
                      "paraphrase" and x.magpie_label == "figurative"}
        intersection = lit_idioms.intersection(fig_idioms)
        sentences = [x for x in sentences if x.idiom in intersection]

    logging.info(f"Processing attention - mode {mode} - {len(sentences)} samples.")

    per_layer = dict()
    for layer in range(6):
        con2idi = defaultdict(list)
        idi2idi = defaultdict(list)
        idi2con = defaultdict(list)

        for sent in sentences:
            # Get the token indices of the idiom and its noun
            nouns_idiom = sent.index_select(1, tags=["NOUN"])
            if len(nouns_idiom) > 1:
                nouns_idiom = random.sample(nouns_idiom, 1)
            all_idiom = sent.index_select(1)
            all_idiom = LT([x for x in all_idiom if x not in nouns_idiom])
            nouns_idiom = LT(nouns_idiom)

            # Get the token indices of tokens in the PIE context
            all_context = sent.index_select(0, neighbours_only=True)
            all_context = LT(all_context)
            nouns_context = sent.index_select(0, tags=["NOUN"], neighbours_only=True)
            nouns_context = LT(nouns_context)

            # Con2idi
            label_pair = (sent.magpie_label, sent.translation_label)
            att = torch.index_select(sent.attention[layer], dim=-2, index=all_context)
            att = torch.mean(torch.index_select(att, dim=-1, index=nouns_idiom))
            con2idi[sent.magpie_label].append(att.item())
            con2idi[sent.translation_label].append(att.item())
            con2idi[label_pair].append(att.item())

            # Idi2idi
            att = torch.index_select(sent.attention[layer], dim=-2, index=all_idiom)
            att = torch.mean(torch.index_select(att, dim=-1, index=nouns_idiom))
            idi2idi[sent.magpie_label].append(att.item())
            idi2idi[sent.translation_label].append(att.item())
            idi2idi[label_pair].append(att.item())

            # Idi2con
            att = torch.index_select(sent.attention[layer], dim=-2, index=all_idiom)
            att = torch.mean(torch.index_select(att, dim=-1, index=nouns_context))
            idi2con[sent.magpie_label].append(att.item())
            idi2con[sent.translation_label].append(att.item())
            idi2con[label_pair].append(att.item())

        per_layer[layer] = {"con2idi": con2idi,
                            "idi2idi": idi2idi,
                            "idi2con": idi2con}

    if mode == "intersection":
        pickle.dump(per_layer, open("data/attention_subset=intersection.pickle", 'wb'))
    elif mode == "identical":
        pickle.dump(per_layer, open("data/attention_subset=identical.pickle", 'wb'))
    else:
        pickle.dump(per_layer, open("data/attention.pickle", 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, choices=["regular", "intersection", "identical"],
        default="regular")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.mode, args.start, args.stop, args.step)
