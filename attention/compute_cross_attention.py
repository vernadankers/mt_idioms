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
import pickle
import logging
logging.basicConfig(level=logging.INFO)

sentences = extract_sentences(
    range(0, 50), use_tqdm=True,
    data_folder="../data/magpie", store_cross_attention=True)
logging.info("Loaded encodings.")

per_setup = dict()
for tags in [[], ["NOUN"], ["NOUN", "VERB", "ADJ", "ADV"]]:
    for neighbourhood in [2, 5, 10]:
        print("Setup: ", tags, neighbourhood)
        per_layer = dict()
        for layer in range(6):
            avg_attention_con2idi = defaultdict(list)
            avg_attention_idi2idi = defaultdict(list)
            avg_attention_idi2con = defaultdict(list)
            avg_attention_idi2eos = defaultdict(list)

            for sent in sentences:
                eos_index = sent.translation.index("</s>")
                translation = sent.translation[:eos_index + 1]
                num_layers, num_heads, output_length, input_length = sent.cross_attention.shape
                align_src2tgt = defaultdict(list)
                align_tgt2src = dict()
                for i in range(eos_index + 1):
                    index = torch.argmax(torch.mean(sent.cross_attention[-1, :, i, :-1], dim=0), dim=-1)
                    align_src2tgt[index.item()].append(i)
                    align_tgt2src[i] = index.item()

                src_indices = sent.index_select(1, tags=tags)
                if src_indices:
                    try:
                        src_indices = torch.LongTensor(src_indices)
                        src_indices2 = torch.LongTensor(
                            sent.index_select(0, tags=tags, neighbours_only=True, neighbourhood=neighbourhood))
                        tgt_indices = [j for i in src_indices for j in align_src2tgt[i.item()]]
                        tgt_indices = torch.LongTensor(sorted(list(set(tgt_indices))))
                        non_tgt_indices = [j for i in src_indices2 for j in align_src2tgt[i.item()]]
                        non_tgt_indices = torch.LongTensor(sorted(list(set(non_tgt_indices))))

                        att = torch.index_select(sent.cross_attention[layer], dim=-2, index=non_tgt_indices)
                        att = torch.mean(torch.index_select(att, dim=-1, index=src_indices))
                        avg_attention_con2idi[sent.magpie_label].append(att.item())
                        avg_attention_con2idi[sent.translation_label].append(att.item())
                        avg_attention_con2idi[(sent.magpie_label, sent.translation_label)].append(att.item())

                        att = torch.index_select(sent.cross_attention[layer], dim=-2, index=tgt_indices)
                        att = torch.mean(torch.index_select(att, dim=-1, index=src_indices))
                        avg_attention_idi2idi[sent.magpie_label].append(att.item())
                        avg_attention_idi2idi[sent.translation_label].append(att.item())
                        avg_attention_idi2idi[(sent.magpie_label, sent.translation_label)].append(att.item())

                        att = torch.index_select(sent.cross_attention[layer], dim=-2, index=tgt_indices)
                        att = torch.mean(torch.index_select(att, dim=-1, index=src_indices2))
                        avg_attention_idi2con[sent.magpie_label].append(att.item())
                        avg_attention_idi2con[sent.translation_label].append(att.item())
                        avg_attention_idi2con[(sent.magpie_label, sent.translation_label)].append(att.item())

                        att = torch.index_select(sent.cross_attention[layer], dim=-2, index=tgt_indices)
                        att = torch.mean(torch.index_select(att, dim=-1, index=torch.LongTensor([len(sent.tokenised_annotation)])))
                        avg_attention_idi2eos[sent.magpie_label].append(att.item())
                        avg_attention_idi2eos[sent.translation_label].append(att.item())
                        avg_attention_idi2eos[(sent.magpie_label, sent.translation_label)].append(att.item())
                    except:
                        logging.info(f"{sent.cross_attention.shape} {len(sent.tokenised_annotation)}")

                # #Idiom and "aligned" translation
                # idiom = [sent.tokenised_sentence.split()[i] for i in src_indices]
                # idiom_translated = [sent.translation[i] for i in tgt_indices]
                # non_idiom_translated = [sent.translation[i] for i in non_tgt_indices]
                # print(idiom_translated, non_idiom_translated)

            per_layer[layer] = {"con2idi": avg_attention_con2idi,
                                "idi2idi": avg_attention_idi2idi,
                                "idi2con": avg_attention_idi2con,
                                "idi2eos": avg_attention_idi2eos}
        per_setup[("_".join(tags) if tags else "all", neighbourhood)] = per_layer


pickle.dump(per_setup, open("cross_attention.pickle", 'wb'))