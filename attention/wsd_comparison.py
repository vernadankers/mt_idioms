import sys
import random
import pickle
import numpy as np
from collections import defaultdict
import torch
sys.path.append('../data/')
from data import extract_sentences


sentences = extract_sentences(
    range(0, 50), use_tqdm=True, data_folder="../data/magpie", store_attention=True)

# Collect all tokenised PIEs
pies = []
for sent in sentences:
    pie_indices = sent.index_select(1, no_subtokens=False)
    pie_indices = range(pie_indices[0], pie_indices[-1] + 1)
    pie = [sent.tokenised_sentence.split()[x] for x in pie_indices]
    pie_pos_tags = [sent.pos_tags[x] for x in pie_indices]
    pies.append(" ".join([sent.tokenised_sentence.split()[x] for x in pie_indices]))
pies = set(pies)

a, b = [], []

# Analyse the encoder's self-attention
per_layer = dict()
for layer in range(6):
    avg_attention = defaultdict(list)
    for sent in sentences:
        pie_indices = sent.index_select(1, no_subtokens=False)
        pie_indices = torch.LongTensor(list(range(pie_indices[0], pie_indices[-1] + 1)))
        pie = [sent.tokenised_sentence.split()[x] for x in pie_indices]
        pie_pos_tags = [sent.pos_tags[x] for x in pie_indices]
        before_context = list(range(pie_indices[0] - 10, pie_indices[0]))
        after_context = list(range(pie_indices[-1] + 1, pie_indices[-1] + 1 + 10))
        pie_context = before_context + after_context
        pie_context = torch.LongTensor([
            x for x in pie_context if 0 <= x < len(sent.tokenised_annotation) and \
            sent.pos_tags[x] == "NOUN"])
        a.append(len(pie_context))
        for i in range(len(sent.pos_tags)):
            non_pie = " ".join(sent.tokenised_sentence.split()[i:i + len(pie_indices)])
            not_idiom = 1 not in sent.tokenised_annotation[i:i + len(pie_indices)]
            equal_pos_tags = pie_pos_tags == sent.pos_tags[i:i + len(pie_indices)]
            if not_idiom and equal_pos_tags and non_pie not in pies:
                att = torch.index_select(sent.attention[layer], dim=-2, index=pie_indices)
                att = torch.mean(torch.index_select(att, dim=-1, index=pie_context))
                avg_attention["attention, pie"].append(att.item())

                before_context = list(range(i - 10, i))
                after_context = list(range(i + len(pie_indices), i + len(pie_indices) + 10))
                non_pie_context = before_context + after_context
                non_pie_context = torch.LongTensor([x for x in non_pie_context if x >= 0 and x < len(sent.tokenised_annotation) and sent.pos_tags[x] == "NOUN"])
                non_pie_indices = torch.LongTensor(list(range(i, i + len(pie_indices))))
                if len(non_pie_context) > 4 and random.random() < 0.5:
                    continue
                b.append(len(non_pie_context))

                att = torch.index_select(sent.attention[layer], dim=-2, index=non_pie_indices)
                att = torch.mean(torch.index_select(att, dim=-1, index=non_pie_context))
                avg_attention["attention, non_pie"].append(att.item())

                break
    per_layer[layer] = avg_attention
pickle.dump(per_layer, open("data/attention_wsd_comparison.pickle", 'wb'))

print(np.mean(a), np.mean(b))
# Analyse the cross-attention
per_layer_cross_attention = dict()
sentences = extract_sentences(
    range(0, 50), use_tqdm=True, data_folder="../data/magpie", store_cross_attention=True)
for layer in range(6):
    avg_attention = defaultdict(list)

    for sent in sentences:
        eos_index = sent.translation.index("</s>")
        translation = sent.translation[:eos_index + 1]
        align_src2tgt = defaultdict(list)
        for i in range(eos_index + 1):
            attention_last_layer = torch.mean(sent.cross_attention[-1, :, i, :-1], dim=0)
            index = torch.argmax(attention_last_layer, dim=-1)
            align_src2tgt[index.item()].append(i)

        pie_indices = sent.index_select(1, tags=["NOUN"])
        if not pie_indices:
            continue
        if len(pie_indices) > 1:
            pie_indices = random.sample(pie_indices, 1)
        tgt_indices = align_src2tgt[pie_indices[0]]
        if not tgt_indices:
            continue

        att = torch.index_select(sent.cross_attention[layer], dim=-2, index=torch.LongTensor(tgt_indices))
        att = torch.mean(torch.index_select(att, dim=-1, index=torch.LongTensor(pie_indices)))
        avg_attention["cross-attention, pie"].append(att.item())

        pie_indices = sent.index_select(0, tags=["NOUN"])
        if not pie_indices:
            continue
        if len(pie_indices) > 1:
            pie_indices = random.sample(pie_indices, 1)
        tgt_indices = align_src2tgt[pie_indices[0]]
        if not tgt_indices:
            continue
        att = torch.index_select(sent.cross_attention[layer], dim=-2, index=torch.LongTensor(tgt_indices))
        att = torch.mean(torch.index_select(att, dim=-1, index=torch.LongTensor(pie_indices)))
        avg_attention["cross-attention, non_pie"].append(att.item())

    per_layer_cross_attention[layer] = avg_attention
pickle.dump(per_layer_cross_attention, open("data/cross_attention_wsd_comparison.pickle", 'wb'))
