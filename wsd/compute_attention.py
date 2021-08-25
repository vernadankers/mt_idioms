from data import extract_sentences
from collections import defaultdict
import pickle
import torch
import logging


def compute_attention(language, label):
    """
    Computes the average attention weights to context for the provided label.

    Args:
        language: de | nl (German / Dutch)
        label: 0 | 1 (unambiguous / ambiguous)
    Returns:
        dictionary with per layer attention weights
    """
    assert language in ["de", "nl"], "Invalid language."
    assert label in [0, 1], "Invalid label provided."

    # Load the data
    sentences = extract_sentences(language, disable_tqdm=True)

    all_attention = defaultdict(list)
    for sent in sentences:
        word_indices = sent.index_select(label, tags=["NOUN"])
        context_indices = sent.index_select([0, 1], tags=["NOUN"])

        for word_index in word_indices:
            # Filter to only include neighbourhood of size 10
            if word_index and context_indices:
                context_subset = [x for x in context_indices if x != word_index
                                  and word_index - 10 < x < word_index + 11]

                if not context_subset:
                    continue

                # First select weights outgoing for current ambiguous word
                context_subset = torch.LongTensor(context_subset)
                attention = torch.index_select(
                    sent.attention,
                    index=torch.LongTensor([word_index]), dim=-2)
                # Then select weights incoming to the context indices
                attention = torch.index_select(
                    attention,
                    index=context_subset, dim=-1)

                # Collect mean weights per layer
                for layer in range(6):
                    mean_attention = torch.mean(attention[layer, :]).item()
                    all_attention[layer].append(mean_attention)
    return dict(all_attention)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Collecting data for German...")
    de_ambiguous_attention = compute_attention(language="de", label=1)
    de_unambiguous_attention = compute_attention(language="de", label=0)

    logging.info("Collecting data for Dutch...")
    nl_ambiguous_attention = compute_attention(language="nl", label=1)
    nl_unambiguous_attention = compute_attention(language="nl", label=0)

    logging.info("Storing to file...")
    pickle.dump({
        ("de", "ambiguous"): de_ambiguous_attention,
        ("de", "unambiguous"): de_unambiguous_attention,
        ("nl", "ambiguous"): nl_ambiguous_attention,
        ("nl", "unambiguous"): nl_unambiguous_attention
    }, open("data/attention.pickle", 'wb'))
