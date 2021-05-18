import sys
from wordfreq import zipf_frequency
sys.path.append('../data/')
sys.path.append('../amnesic_probing/')
from probing import set_seed
import pickle
import os
import numpy as np
import torch
import math
import random
import seaborn as sns
import cca_core
import scipy.stats

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from data import extract_sentences
from collections import defaultdict, Counter

sns.set_context("talk")

def compute_cosine_sim(data1, data2, cca_results, cca=True):
    k = 512
    cacts1 = data1 - np.mean(data1, axis=1, keepdims=True)
    cacts2 = data2 - np.mean(data2, axis=1, keepdims=True)

    data1 = np.dot(np.dot(cca_results["full_coef_x"][:k].T, np.dot(cca_results["full_coef_x"][:k], cca_results["full_invsqrt_xx"])),
                   cacts1)
    data2 = np.dot(np.dot(cca_results["full_coef_y"][:k].T, np.dot(cca_results["full_coef_y"][:k], cca_results["full_invsqrt_yy"])),
                   cacts2)
    data1 = data1 - np.mean(data1, axis=1, keepdims=True)
    data2 = data2 - np.mean(data2, axis=1, keepdims=True)
    cos = torch.nn.CosineSimilarity(dim=1)
    cos = cos(torch.FloatTensor(data1), torch.FloatTensor(data2))
    print(cos.shape)
    return torch.mean(cos).item()

# Load all hidden representations of idioms
samples3 = extract_sentences(range(1000, 1727), use_tqdm=True, store_hidden_states=True, data_folder="../data/magpie")
samples2 = extract_sentences(range(0, 1000), use_tqdm=True, store_hidden_states=True, data_folder="../data/magpie")

vocab = []
for s in samples2:
    vocab.extend(s.tokenised_sentence.split())
vocab = Counter(vocab)
vocab = list(set(x for x in vocab if vocab[x] >= 16 and vocab[x] <= 100))
print(len(vocab))


vocab_size = {0: (320, 16), 1: (423, 12), 2: (640, 8), 3: (853, 6), 4: (1280, 4)}
vocab_sets = dict()
for cat in [0, 1, 2, 3, 4]:
    random.shuffle(vocab)
    vocab_sets[cat] = vocab[:vocab_size[cat][0]]
    

all_coefs = defaultdict(list)
for seed in [1]:
    random.seed(seed)
    random.shuffle(samples2)
    print("Collecting hidden states.")
    freq_subsets = defaultdict(lambda: defaultdict(list))
    vocab_freqs = defaultdict(lambda: defaultdict(Counter))
    for cat in [0, 1, 2, 3, 4]:
        max_freq = vocab_size[cat][1]
        for layer in range(7):
            for s in samples2:
                for word, hidden_state in zip(s.tokenised_sentence.split(), s.hidden_states[layer]):
                    if word in vocab_sets[cat] and vocab_freqs[layer][cat][word] < max_freq:
                        freq_subsets[layer][cat].append(hidden_state)
                        vocab_freqs[layer][cat][word] += 1

    print("Doing SVCCA.")
    for category in [0, 1, 2, 3, 4]:
        coefs = []
        for i in range(1, 6):
            layer1 = torch.stack(freq_subsets[i][category], dim=0).transpose(0, 1).numpy()
            layer2 = torch.stack(freq_subsets[i + 1][category], dim=0).transpose(0, 1).numpy()
            print(layer1.shape)
            cca = cca_core.get_cca_similarity(layer1[:, :10000], layer2[:, :10000], epsilon=1e-6)
            coefs.append(np.mean(cca["cca_coef1"])) #coefs.append(cca) # 
        all_coefs[category].append(coefs)
        
fig = plt.figure(figsize=(4.5, 4.5))
for category, label in [(0, "320 x 16"), (1, "423 x 12"), (2, "640 x 8"), (3, "853 x 6"), (4, "1280 x 4")]:
    colours = sns.color_palette("viridis", 8)
    mean = np.mean(all_coefs[category], axis=0)
    print(mean)
    std = np.std(all_coefs[category], axis=0)
    sns.scatterplot(x=[1, 2, 3, 4, 5], y=mean, color=colours[category])
    ax = sns.lineplot(x=[1, 2, 3, 4, 5], y=mean, color=colours[category], alpha=0.5)
    plt.fill_between([1, 2, 3, 4, 5],
                     [x - y for x, y in zip(mean, std)],
                     [x + y for x, y in zip(mean, std)], color=colours[category], alpha=0.2)
plt.xlabel("layers")
plt.ylabel("SVCCA")
plt.xticks([1, 2, 3, 4, 5], ["1-2", "2-3", "3-4", "4-5", "5-6"])

plt.savefig("figures/svcca.pdf", bbox_inches="tight")
plt.show()


cca_results = []
random.shuffle(samples3)
for i in range(1, 7):
    layer1 = []
    layer2 = []
    for s in samples3:
        for word, hid in zip(s.tokenised_sentence.split(), s.hidden_states[i]):
            layer1.append(hid.tolist())
        for word, hid in zip(s.tokenised_sentence.split(), s.hidden_states[i - 1]):
            layer2.append(hid.tolist())

    layer1 = torch.FloatTensor(layer1).transpose(0, 1).numpy()
    layer2 = torch.FloatTensor(layer2).transpose(0, 1).numpy()
    cca = cca_core.get_cca_similarity(layer1[:, :100000], layer2[:, :100000], epsilon=1e-6)
    cca_results.append(cca)


vocab_size = {0: (320, 16), 1: (423, 12), 2: (640, 8), 3: (853, 6), 4: (1280, 4)}
vocab_sets = dict()
for cat in [0, 1, 2, 3, 4]:
    random.shuffle(vocab)
    vocab_sets[cat] = vocab[:vocab_size[cat][0]]

all_coefs = defaultdict(list)

for seed in [1]:
    random.seed(seed)
    random.shuffle(samples2)
    print("Collecting hidden states.")
    freq_subsets = defaultdict(lambda: defaultdict(list))
    vocab_freqs = defaultdict(lambda: defaultdict(Counter))
    for cat in [0, 1, 2, 3, 4]:
        max_freq = vocab_size[cat][1]
        for layer in range(7):
            for s in samples2:
                for word, hidden_state in zip(s.tokenised_sentence.split(), s.hidden_states[layer]):
                    if word in vocab_sets[cat] and vocab_freqs[layer][cat][word] < max_freq:
                        freq_subsets[layer][cat].append(hidden_state)
                        vocab_freqs[layer][cat][word] += 1

    print("Doing SVCCA.")
    for category in [0, 1, 2, 3, 4]:
        coefs = []
        for i in range(2, 7):
            layer1 = torch.stack(freq_subsets[i][category], dim=0).transpose(0, 1).numpy()
            layer2 = torch.stack(freq_subsets[i - 1][category], dim=0).transpose(0, 1).numpy()
            print(layer1.shape)
            coefs.append(compute_cosine_sim(layer1, layer2, cca_results[i - 1]))
        all_coefs[category].append(coefs)


fig = plt.figure(figsize=(4.5, 4.5))
for category, label in [(0, "320 x 16"), (1, "423 x 12"), (2, "640 x 8"), (3, "853 x 6"), (4, "1280 x 4")]:
    colours = sns.color_palette("viridis", 8)
    mean = np.mean(all_coefs[category], axis=0)
    std = np.std(all_coefs[category], axis=0)
    sns.scatterplot(x=[1, 2, 3, 4, 5], y=mean, color=colours[category], label=label)
    ax = sns.lineplot(x=[1, 2, 3, 4, 5], y=mean, color=colours[category], alpha=0.5)
    plt.fill_between([1, 2, 3, 4, 5],
                     [x - y for x, y in zip(mean, std)],
                     [x + y for x, y in zip(mean, std)], color=colours[category], alpha=0.2)
plt.xlabel("layers")
plt.ylabel("Two-step SVCCA")
plt.xticks([1, 2, 3, 4, 5], ["1-2", "2-3", "3-4", "4-5", "5-6"])
plt.legend(bbox_to_anchor=(0.95, 1.05), frameon=False)
plt.savefig("figures/two_step_svcca.pdf", bbox_inches="tight")
plt.show()