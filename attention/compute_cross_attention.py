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
    sentences = extract_sentences(
        range(start, stop, step), use_tqdm=False,
        data_folder="../data/magpie", store_cross_attention=True)

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

    logging.info(f"Processing cross-attention - mode {mode} - {len(sentences)} samples.")

    per_layer = dict()
    for layer in range(6):
        idi2idi = defaultdict(list)
        idi2eos = defaultdict(list)
        idi2con = defaultdict(list)

        for sent in sentences:
            # Compute which words align to which
            eos_index = sent.translation.index("</s>")
            align_src2tgt = defaultdict(list)
            for i in range(eos_index + 1):
                if not  u'▁' in sent.translation[i]:
                    continue
                att_last_layer = torch.mean(sent.cross_attention[-1, :, i, :-1], dim=0)
                index = torch.argmax(att_last_layer, dim=-1)
                align_src2tgt[index.item()].append(i)

            # Get src PIE indices
            src_pie_indices = sent.index_select(1, tags=["NOUN"])
            src_other_indices = [x for x in sent.index_select(1, tags=[])
                                 if x not in src_pie_indices]
            if not src_pie_indices:
                continue
            if len(src_pie_indices) > 1:
                src_pie_indices = random.sample(src_pie_indices, 1)
            # Now align to the words that the PIE terms translated in
            tgt_indices = [j for i in src_pie_indices for j in align_src2tgt[i]]
            tgt_indices = LT(sorted(list(set(tgt_indices))))

            # Tgt PIE to src PIE attention
            att = torch.index_select(sent.cross_attention[layer], dim=-2, index=tgt_indices)
            att = torch.mean(torch.index_select(att, dim=-1, index=LT(src_pie_indices)))
            idi2idi[sent.magpie_label].append(att.item())
            idi2idi[sent.translation_label].append(att.item())
            idi2idi[(sent.magpie_label, sent.translation_label)].append(att.item())

            # Tgt PIE to src EOS attention
            att = torch.index_select(sent.cross_attention[layer], dim=-2, index=tgt_indices)
            att = torch.mean(torch.index_select(
                att, dim=-1, index=LT([len(sent.tokenised_annotation)])))
            idi2eos[sent.magpie_label].append(att.item())
            idi2eos[sent.translation_label].append(att.item())
            idi2eos[(sent.magpie_label, sent.translation_label)].append(att.item())

            # Tgt PIE to non-PIE in the source attention
            att = torch.index_select(sent.cross_attention[layer], dim=-2, index=tgt_indices)
            att = torch.mean(torch.index_select(att, dim=-1, index=LT(src_other_indices)))
            idi2con[sent.magpie_label].append(att.item())
            idi2con[sent.translation_label].append(att.item())
            idi2con[(sent.magpie_label, sent.translation_label)].append(att.item())

            per_layer[layer] = {"idi2idi": idi2idi, "idi2eos": idi2eos,
                                "idi2con": idi2con}

    if mode == "intersection":
        pickle.dump(per_layer, open("data/cross_attention_subset=intersection.pickle", 'wb'))
    elif mode == "identical":
        pickle.dump(per_layer, open("data/cross_attention_subset=identical.pickle", 'wb'))
    else:
        pickle.dump(per_layer, open("data/cross_attention.pickle", 'wb'))


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
