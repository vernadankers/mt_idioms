import sys
import random
import pickle
from collections import defaultdict
import torch
from torch import LongTensor as LT
sys.path.append('../data/')
from data import extract_sentences


def get_attention(attention, from_indices, to_indices):
    """Extract attention from ... to ... and average over heads."""
    att = torch.index_select(attention, dim=-2, index=LT(from_indices))
    att = torch.mean(torch.index_select(att, dim=-1, index=LT(to_indices)))
    return att.item()


def collect_attention():
    """Encoder self-attention for PIEs and non-PIEs."""
    sentences = extract_sentences(
        range(0, 1727), use_tqdm=True, data_folder="../data/magpie", store_attention=True)

    # Collect all tokenised PIEs
    pies = []
    for sent in sentences:
        pie_indices = sent.index_select(1, no_subtokens=False)
        pie_indices = range(pie_indices[0], pie_indices[-1] + 1)
        pies.append(" ".join([sent.tokenised_sentence.split()[x] for x in pie_indices]))
    pies = set(pies)

    # Analyse the encoder's self-attention
    per_layer = dict()
    for layer in range(6):
        avg_attention = defaultdict(list)
        for sent in sentences:
            pie_indices = sent.index_select(1, no_subtokens=False)
            pie_indices = list(range(pie_indices[0], pie_indices[-1] + 1))
            pie_pos_tags = [sent.pos_tags[x] for x in pie_indices]
            before_context = list(range(pie_indices[0] - 10, pie_indices[0]))
            after_context = list(range(pie_indices[-1] + 1, pie_indices[-1] + 1 + 10))
            pie_context = [
                x for x in before_context + after_context \
                if 0 <= x < len(sent.tokenised_annotation) and sent.pos_tags[x] == "NOUN"]

            for i in range(len(sent.pos_tags)):
                non_pie = " ".join(sent.tokenised_sentence.split()[i:i + len(pie_indices)])
                not_idiom = 1 not in sent.tokenised_annotation[i:i + len(pie_indices)]
                equal_pos_tags = pie_pos_tags == sent.pos_tags[i:i + len(pie_indices)]

                if not_idiom and equal_pos_tags and non_pie not in pies:
                    avg_attention["attention, pie"].append(
                        get_attention(sent.attention[layer], pie_indices, pie_context))

                    before_context = list(range(i - 10, i))
                    after_context = list(range(i + len(pie_indices), i + len(pie_indices) + 10))
                    non_pie_context = [
                        x for x in before_context + after_context if 0 <= x < \
                        len(sent.tokenised_annotation) and sent.pos_tags[x] == "NOUN"]
                    non_pie_indices = list(range(i, i + len(pie_indices)))
                    avg_attention["attention, non_pie"].append(
                        get_attention(sent.attention[layer], non_pie_indices, non_pie_context))
                    break
        per_layer[layer] = avg_attention
    pickle.dump(per_layer, open("data/attention_wsd_comparison.pickle", 'wb'))


def collect_cross_attention():
    """Cross-attention PIE src to PIE tgt and non-PIE src to non-PIE tgt."""
    # Analyse the cross-attention
    per_layer_cross_attention = dict()
    sentences = extract_sentences(
        range(0, 1727), use_tqdm=True,
        data_folder="../data/magpie", store_cross_attention=True)
    for layer in range(6):
        avg_attention = defaultdict(list)
        for sent in sentences:

            # Get the sentence alignment
            align_src2tgt = defaultdict(list)
            for i in range(sent.translation.index("</s>") + 1):
                last_layer = torch.mean(sent.cross_attention[-1, :, i, :-1], dim=0)
                index = torch.argmax(last_layer, dim=-1)
                align_src2tgt[index.item()].append(i)

            # Get the PIE and non-PIE indices
            pie_src = sent.index_select(1, tags=["NOUN"])
            non_pie_src = sent.index_select(0, tags=["NOUN"])
            if not pie_src or not non_pie_src:
                continue

            # Sample 1 to measure 1 - 1 attention
            if len(pie_src) > 1:
                pie_src = random.sample(pie_src, 1)
            if len(non_pie_src) > 1:
                non_pie_src = random.sample(non_pie_src, 1)

            # Find tgt indices that align to the (non-)PIE
            pie_tgt = align_src2tgt[pie_src[0]]
            non_pie_tgt = align_src2tgt[non_pie_src[0]]

            if not pie_tgt or not non_pie_tgt:
                continue

            avg_attention["cross-attention, pie"].append(
                get_attention(sent.cross_attention[layer], pie_tgt, pie_src))
            avg_attention["cross-attention, non_pie"].append(
                get_attention(sent.cross_attention[layer], non_pie_tgt, non_pie_src))

        per_layer_cross_attention[layer] = avg_attention
    pickle.dump(per_layer_cross_attention,
                open("data/cross_attention_wsd_comparison.pickle", 'wb'))


if __name__ == "__main__":
    collect_attention()
    collect_cross_attention()
