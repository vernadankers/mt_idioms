import sys

sys.path.append('../data/')
sys.path.append('../probing/')

from data import extract_sentences, Sentence
from collections import defaultdict
import torch
import numpy as np
import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt
import random
import argparse
import logging
import pickle
logging.basicConfig(level=logging.INFO)

mode = "regular"

########################### Step 1: load the data
sentences = extract_sentences(
    range(0, 1727), use_tqdm=True, data_folder="../data/magpie", store_attention=True)

if mode == "identical":
    sentences = [x for x in sentences if x.variant == "identical"]
elif mode == "intersection":
    literal_idioms = {x.idiom for x in sentences if x.translation_label == "word-by-word" and x.magpie_label == "literal"}
    figurative_idioms = {x.idiom for x in sentences if x.translation_label == "paraphrase" and x.magpie_label == "figurative"}
    intersection = literal_idioms.intersection(figurative_idioms)
    sentences = [x for x in sentences if x.idiom in intersection]

logging.info(f"{len(sentences)} sentences available.")

logging.info("Loaded encodings.")

per_setup = dict()
for tags in [["NOUN"]]: #, ["NOUN", "VERB", "ADJ", "ADV"]]:
    for neighbourhood in [10]:
        logging.info(f"Setup: {'_'.join(tags) if tags else 'all'} {neighbourhood}")

        per_layer = dict()
        for layer in range(6):
            avg_attention_con2idi = defaultdict(list)
            avg_attention_idi2idi = defaultdict(list)
            avg_attention_idi2con = defaultdict(list)
            avg_attention_idinoun2con = defaultdict(list)
            avg_attention_noun2con = defaultdict(list)

            for sent in sentences:
                num_layers, num_heads, output_length, input_length = sent.attention.shape

                
                idiom_indices = sent.index_select(1, tags=tags)
                if len(idiom_indices) > 1:
                    idiom_indices = random.sample(idiom_indices, 1)
                all_idiom_indices = torch.LongTensor([x for x in sent.index_select(1) if x not in idiom_indices])
                idiom_indices = torch.LongTensor(idiom_indices)
                all_context_indices = torch.LongTensor(
                    sent.index_select(0, neighbours_only=True, neighbourhood=neighbourhood))
                context_indices = torch.LongTensor(
                    sent.index_select(0, tags=tags, neighbours_only=True, neighbourhood=neighbourhood))

                att = torch.index_select(sent.attention[layer], dim=-2, index=all_context_indices)
                att = torch.mean(torch.index_select(att, dim=-1, index=idiom_indices))
                avg_attention_con2idi[sent.magpie_label].append(att.item())
                avg_attention_con2idi[sent.translation_label].append(att.item())
                avg_attention_con2idi[(sent.magpie_label, sent.translation_label)].append(att.item())

                att = torch.index_select(sent.attention[layer], dim=-2, index=all_idiom_indices)
                att = torch.mean(torch.index_select(att, dim=-1, index=idiom_indices))
                avg_attention_idi2idi[sent.magpie_label].append(att.item())
                avg_attention_idi2idi[sent.translation_label].append(att.item())
                avg_attention_idi2idi[(sent.magpie_label, sent.translation_label)].append(att.item())

                att = torch.index_select(sent.attention[layer], dim=-2, index=all_idiom_indices)
                att = torch.mean(torch.index_select(att, dim=-1, index=context_indices))
                avg_attention_idi2con[sent.magpie_label].append(att.item())
                avg_attention_idi2con[sent.translation_label].append(att.item())
                avg_attention_idi2con[(sent.magpie_label, sent.translation_label)].append(att.item())

                idiom_noun_indices = sent.index_select(1, tags=["NOUN"])
                if len(idiom_noun_indices) > 1:
                    idiom_noun_indices = random.sample(idiom_noun_indices, 1)
                if idiom_noun_indices:
                    idiom_noun_indices = torch.LongTensor(idiom_noun_indices)
                    idiom_context_indices = [
                        x for x in sent.index_select([0]) \
                        if (idiom_noun_indices[0] - 10 <= x and x <= idiom_noun_indices[0] - 5) or \
                        (idiom_noun_indices[0] + 5 <= x and x <= idiom_noun_indices[0] + 10)]
                    idiom_context_indices = torch.LongTensor(idiom_context_indices)
                    att = torch.index_select(sent.attention[layer], dim=-2, index=idiom_noun_indices)
                    att = torch.mean(torch.index_select(att, dim=-1, index=idiom_context_indices))
                    avg_attention_idinoun2con[sent.magpie_label].append(att.item())
                    avg_attention_idinoun2con[sent.translation_label].append(att.item())
                    avg_attention_idinoun2con[(sent.magpie_label, sent.translation_label)].append(att.item())

                noun_indices = sent.index_select(0, tags=["NOUN"])
                if len(noun_indices) > 1:
                    noun_indices = random.sample(noun_indices, 1)
                if noun_indices:
                    noun_indices = torch.LongTensor(noun_indices)
                    context_indices = [
                        x for x in sent.index_select([0]) \
                        if (noun_indices[0] - 10 <= x and x <= noun_indices[0] - 5) or \
                        (noun_indices[0] + 5 <= x and x <= noun_indices[0] + 10)]
                    context_indices = torch.LongTensor(context_indices)
                    att = torch.index_select(sent.attention[layer], dim=-2, index=noun_indices)
                    att = torch.mean(torch.index_select(att, dim=-1, index=context_indices))
                    avg_attention_noun2con[sent.magpie_label].append(att.item())
                    avg_attention_noun2con[sent.translation_label].append(att.item())
                    avg_attention_noun2con[(sent.magpie_label, sent.translation_label)].append(att.item())

            per_layer[layer] = {"con2idi": avg_attention_con2idi,
                                "idi2idi": avg_attention_idi2idi,
                                "idi2con": avg_attention_idi2con,
                                "idinoun2con": avg_attention_idinoun2con,
                                "noun2con": avg_attention_noun2con}
        per_setup[("_".join(tags) if tags else "all", neighbourhood)] = per_layer

if mode == "intersection":
    pickle.dump(per_setup, open("attention_subset=intersection.pickle", 'wb'))
elif mode == "identical":
    pickle.dump(per_setup, open("attention_subset=identical.pickle", 'wb'))
else:
    pickle.dump(per_setup, open("attention.pickle", 'wb'))
