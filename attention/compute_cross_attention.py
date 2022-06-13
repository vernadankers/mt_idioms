import sys
sys.path.append('../data/')
from data import extract_sentences
import os
from collections import defaultdict
import random
import argparse
import logging
import pickle
import torch
from classifier import Classifier
from transformers import MarianTokenizer
from torch import LongTensor as LT
random.seed(1)


def is_sub(sub, lst):
    ln = len(sub)
    for i in range(len(lst) - ln + 1):
        if all(sub[j] == lst[i+j] for j in range(ln)):
            return True
    return False


def main(mode, start, stop, step, language, use_precomputed_alignments=False, folder="data"):
    src_not_found = []

    if use_precomputed_alignments:
        precomputed_alignments = pickle.load(
            open(f"../data/magpie/{language}/alignments.pickle", 'rb'))

    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)

    sentences = extract_sentences(
        range(start, stop, step), classifier, tokenizer, use_tqdm=False,
        data_folder=f"../data/magpie/{language}", store_cross_attention=True)

    # Restrict the data used to...
    # - identical are matches labelled as identical by MAGPIE
    # - intersection are PIEs that are both in fig-par and lit-wfw
    if mode == "identical":
        sentences = [x for x in sentences if x.variant == "identical"]
    elif mode == "intersection":
        lit_idioms = {x.idiom for x in sentences if x.translation_label
                      == "word-by-word" and x.magpie_label == "literal"}
        fig_idioms = {x.idiom for x in sentences if x.translation_label
                      == "paraphrase" and x.magpie_label == "figurative"}
        intersection = lit_idioms.intersection(fig_idioms)
        sentences = [x for x in sentences if x.idiom in intersection]
    elif mode == "short":
        sentences = [
            x for x in sentences
            if x.tokenised_annotation.count(1) == 3 and \
               (is_sub([1, 0, 1, 1], x.tokenised_annotation) or is_sub([1, 1, 0, 1], x.tokenised_annotation))]

    logging.info(
        f"Processing cross-attention - mode {mode} - {len(sentences)} samples.")
    logging.info(f"Translation 1: {sentences[0].translation}")

    per_layer = dict()
    for layer in range(6):
        idi2idi = defaultdict(list)
        idi2eos = defaultdict(list)
        idi2con = defaultdict(list)

        for counter, sent in enumerate(sentences):
            if counter % 500 == 0:
                logging.info(f"...layer {layer} - sentence {counter}...")
            # Compute which words align to which
            eos_index = sent.translation.index("</s>")
            if not use_precomputed_alignments:
                align_src2tgt = defaultdict(list)
                for i in range(eos_index + 1):
                    if not u'▁' in sent.translation[i]:
                        continue
                    att_last_layer = torch.mean(
                        torch.mean(sent.cross_attention[:, :, i, :-1], dim=0), dim=0)
                    index = torch.argmax(att_last_layer, dim=-1)
                    align_src2tgt[index.item()].append(i)
            else:
                src = ''.join(sent.tokenised_sentence).replace(' ', '').replace(u'▁', ' ').strip()
                if src in precomputed_alignments:
                    align_src2tgt = precomputed_alignments[src]
                else:
                    src_not_found.append(src)
                    continue

            # Get src PIE indices
            src_pie_indices = sent.index_select(1, tags=["NOUN"])
            src_other_indices = [x for x in sent.index_select(1, tags=[])
                                 if x not in src_pie_indices]
            if not src_pie_indices:
                continue
            if len(src_pie_indices) > 1:
                src_pie_indices = random.sample(src_pie_indices, 1)
            # Now align to the words that the PIE terms translated in
            tgt_indices = [
                j for i in src_pie_indices for j in align_src2tgt.get(i, [])]
            tgt_indices = LT(sorted(list(set(tgt_indices))))

            # Tgt PIE to src PIE attention
            att = torch.index_select(
                sent.cross_attention[layer], dim=-2, index=tgt_indices)
            att = torch.mean(torch.index_select(
                att, dim=-1, index=LT(src_pie_indices)))
            idi2idi[sent.magpie_label].append(att.item())
            idi2idi[sent.translation_label].append(att.item())
            idi2idi[(sent.magpie_label, sent.translation_label)
                    ].append(att.item())

            # Tgt PIE to src EOS attention
            att = torch.index_select(
                sent.cross_attention[layer], dim=-2, index=tgt_indices)
            att = torch.mean(torch.index_select(
                att, dim=-1, index=LT([len(sent.tokenised_annotation)])))
            idi2eos[sent.magpie_label].append(att.item())
            idi2eos[sent.translation_label].append(att.item())
            idi2eos[(sent.magpie_label, sent.translation_label)
                    ].append(att.item())

            # Tgt PIE to non-PIE in the source attention
            att = torch.index_select(
                sent.cross_attention[layer], dim=-2, index=tgt_indices)
            att = torch.mean(torch.index_select(
                att, dim=-1, index=LT(src_other_indices)))
            idi2con[sent.magpie_label].append(att.item())
            idi2con[sent.translation_label].append(att.item())
            idi2con[(sent.magpie_label, sent.translation_label)
                    ].append(att.item())

            per_layer[layer] = {"idi2idi": idi2idi, "idi2eos": idi2eos,
                                "idi2con": idi2con}

    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(f"{folder}/{language}"):
        os.mkdir(f"{folder}/{language}")
    if mode == "intersection":
        pickle.dump(per_layer, open(
            f"{folder}/{language}/cross_attention_subset=intersection_eflomal={use_precomputed_alignments}.pickle", 'wb'))
    elif mode == "identical":
        pickle.dump(per_layer, open(
            f"{folder}/{language}/cross_attention_subset=identical_eflomal={use_precomputed_alignments}.pickle", 'wb'))
    elif mode == "short":
        pickle.dump(per_layer, open(
            f"{folder}/{language}/cross_attention_subset=short_eflomal={use_precomputed_alignments}.pickle", 'wb'))
    else:
        pickle.dump(per_layer, open(
            f"{folder}/{language}/cross_attention_eflomal={use_precomputed_alignments}.pickle", 'wb'))

    print(src_not_found)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--language", type=str, default="nl")
    parser.add_argument("--use_precomputed_alignments", action="store_true")
    parser.add_argument("--folder", type=str, default="data")
    parser.add_argument(
        "--mode", type=str, choices=["regular", "intersection", "identical", "short"],
        default="regular")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.mode, args.start, args.stop, args.step, args.language,
         args.use_precomputed_alignments, args.folder)
