import sys
sys.path.append('../data/')
sys.path.append("../probing")
from train_probes import set_seed
import random
import torch
from classifier import Classifier
from data import extract_sentences
import numpy as np
from collections import defaultdict
import logging
logging.getLogger().setLevel(logging.INFO)
import torch
from sklearn.metrics import f1_score
from debias import get_debiasing_projection, SKlearnClassifier, load_data
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--l", type=int, default=200)
    parser.add_argument("--setup", type=str, choices=["attention", "hidden"])
    parser.add_argument("--target", type=str, choices=["both", "magpie"])
    parser.add_argument("--average_pie", action="store_true")
    args = parser.parse_args()
    print(vars(args))

    set_seed(333)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.m, args.n, args.k):
        if (i + 1) % 50 == 0:
            logging.info(f"Sample {i/args.k:.0f} / {(args.n-args.m)/args.k:.0f}")
        samples = extract_sentences(
            [i], use_tqdm=False, store_hidden_states=True, store_attention_query=True)
        for s in samples:
            if args.target == "both":
                if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                    s.label = 1
                elif s.translation_label == "word-by-word" and s.magpie_label == "literal":
                    s.label = 0
                else:
                    s.label = None
            else:
                if s.magpie_label == "figurative":
                    s.label = 1
                elif s.magpie_label == "literal":
                    s.label = 0
                else:
                    s.label = None

        samples = [s for s in samples if s.label is not None]
        if samples:
            data[i] = samples

    indices = list(data.keys())
    random.shuffle(indices)

    n = int(len(indices)/5)
    fold_1 = [s for i in indices[:n] for s in data[i]]
    fold_2 = [s for i in indices[n:n*2] for s in data[i]]
    fold_3 = [s for i in indices[n*2:n*3] for s in data[i]]
    fold_4 = [s for i in indices[n*3:n*4] for s in data[i]]
    fold_5 = [s for i in indices[n*4:] for s in data[i]]

    train_1 = fold_2 + fold_3 + fold_4 + fold_5
    test_1 = fold_1

    train_2 = fold_1 + fold_3 + fold_4 + fold_5
    test_2 = fold_2

    train_3 = fold_1 + fold_2 + fold_4 + fold_5
    test_3 = fold_3

    train_4 = fold_1 + fold_2 + fold_3 + fold_5
    test_4 = fold_4

    train_5 = fold_1 + fold_2 + fold_3 + fold_4
    test_5 = fold_5

    folds = [(train_1, test_1), (train_2, test_2), (train_3, test_3),
             (train_4, test_4), (train_5, test_5)]

    for layer in range(6 if args.setup == "attention" else 7):
        f1s, baseline_f1s = [], []
        for train, test in folds:
            X_train, Y_train, median_freq = load_data(
                train, layer, args.setup == "attention", False,
                average_pie=args.average_pie)
            X_dev, Y_dev, _ = load_data(
                test, layer, args.setup == "attention", False,
                median_freq, average_pie=args.average_pie)
            clf = SKlearnClassifier(LogisticRegression(**{"max_iter": 5000, "random_state": 1}))
            f1_macro = clf.train_network(X_train, Y_train, X_dev, Y_dev)

            f1s.append(f1_macro)
            baseline_f1s.append(f1_score(Y_dev, [random.choice(Y_train) for _ in Y_dev], average="macro"))
        logging.info(f"{layer} {np.mean(f1s):.3f} {np.std(f1s)}, {np.mean(baseline_f1s):.3f}")
