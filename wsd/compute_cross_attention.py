from data import extract_sentences
from collections import defaultdict
import torch
import logging
import pickle


def compute_cross_attention(language, label):
    assert language in ["de", "nl"], "Invalid language."
    assert label in [0, 1], "Invalid label provided."

    sentences = extract_sentences(language, disable_tqdm=True)
    all_attention = defaultdict(list)

    for sent in sentences:
        eos_index = sent.translation.index("</s>")

        # Collect which tokens align to each other in the last attention layer
        align_src2tgt = defaultdict(list)
        for i in range(eos_index + 1):
            index = torch.argmax(torch.mean(
                sent.cross_attention[-1, :, i, :-1], dim=0), dim=-1)
            align_src2tgt[index.item()].append(i)
        src_indices = sent.index_select(label, tags=["NOUN"])

        for focus_index in src_indices:
            # Measure attention from src "focus" index to its aligned tokens
            tgt_indices = [j for j in align_src2tgt[focus_index]]
            # Select weights outgoing from those aligned target tokens
            attention = torch.index_select(
                sent.cross_attention,
                index=torch.LongTensor(tgt_indices), dim=-2)
            # ...to the focus index
            attention = torch.index_select(
                attention,
                index=torch.LongTensor([focus_index]), dim=-1)

            if len(attention) == 0:
                continue

            # Store attention per layer
            for layer in range(6):
                mean = torch.mean(attention[layer, :]).item()
                all_attention[layer].append(mean)
    return all_attention


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Collecting data for German...")
    de_ambiguous_attention = compute_cross_attention(language="de", label=1)
    de_unambiguous_attention = compute_cross_attention(language="de", label=0)

    logging.info("Collecting data for Dutch...")
    nl_ambiguous_attention = compute_cross_attention(language="nl", label=1)
    nl_unambiguous_attention = compute_cross_attention(language="nl", label=0)

    logging.info("Storing to file...")
    pickle.dump({
        ("de", "ambiguous"): de_ambiguous_attention,
        ("de", "unambiguous"): de_unambiguous_attention,
        ("nl", "ambiguous"): nl_ambiguous_attention,
        ("nl", "unambiguous"): nl_unambiguous_attention
    }, open("data/cross_attention.pickle", 'wb'))
