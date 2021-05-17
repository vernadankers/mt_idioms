import sys
sys.path.append('../data/')
import random
import torch
import pickle
from data import extract_sentences
import numpy as np
from collections import defaultdict
import logging
logging.getLogger().setLevel(logging.INFO)
import torch
from sklearn.metrics import f1_score
from debias import get_debiasing_projection, SKlearnClassifier, load_data
from sklearn.linear_model import LogisticRegression


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    logging.info(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--setup", type=str, choices=["attention", "hidden"])
    parser.add_argument("--target", type=str, choices=["both", "magpie"])
    parser.add_argument("--average_pie", action="store_true")
    parser.add_argument("--split_by_idiom", action="store_true")
    args = parser.parse_args()
    print(vars(args))

    set_seed(333)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.start, args.stop, args.step):
        if (i + 1) % 50 == 0:
            logging.info(f"Loading idiom {i/args.step:.0f} / {(args.stop-args.start)/args.step:.0f}")
        samples = extract_sentences(
            [i], use_tqdm=False, store_hidden_states=args.setup == "hidden",
            store_attention_query=args.setup == "attention")
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

    if args.split_by_idiom:
        n = int(len(indices)/5)
        fold_1 = [s for i in indices[:n] for s in data[i]]
        fold_2 = [s for i in indices[n:n*2] for s in data[i]]
        fold_3 = [s for i in indices[n*2:n*3] for s in data[i]]
        fold_4 = [s for i in indices[n*3:n*4] for s in data[i]]
        fold_5 = [s for i in indices[n*4:] for s in data[i]]

    else:
        samples = [s for i in indices for s in data[i]]
        n = int(len(samples)/5)
        random.shuffle(samples)
        fold_1 = [s for s in samples[:n]]
        fold_2 = [s for s in samples[n:n*2]]
        fold_3 = [s for s in samples[n*2:n*3]]
        fold_4 = [s for s in samples[n*3:n*4]]
        fold_5 = [s for s in samples[n*4:]]

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

    scores_dict = defaultdict(lambda: defaultdict(list))
    for layer in range(6 if args.setup == "attention" else 7):
        f1s_train, f1s_test, baseline_f1s = [], [], []
        for train, test in folds:
            train_sentences = [s.sentence+s.idiom for s in train]
            test_sentences = [s.sentence+s.idiom for s in test]
            assert not set(train_sentences).intersection(set(test_sentences)), \
                len(set(train_sentences).intersection(set(test_sentences)))

            X_train, Y_train, median_freq = load_data(
                train, layer, args.setup == "attention", False,
                average_pie=args.average_pie)
            X_dev, Y_dev, _ = load_data(
                test, layer, args.setup == "attention", False,
                median_freq, average_pie=args.average_pie)
            clf = SKlearnClassifier(LogisticRegression(**{"max_iter": 5000, "random_state": 1}))
            f1_train, f1_test = clf.train_network(X_train, Y_train, X_dev, Y_dev)
            f1_baseline = f1_score(Y_dev, [random.choice(Y_train) for _ in Y_dev], average="macro")

            scores_dict[layer]["f1_train"].append(f1_train)
            scores_dict[layer]["f1_test"].append(f1_test)
            scores_dict[layer]["f1_baseline"].append(f1_baseline)

            f1s_train.append(f1_train)
            f1s_test.append(f1_test)
            baseline_f1s.append(f1_baseline)


        logging.info(f"{layer}, {np.mean(f1s_train):.3f} {np.std(f1s_train):.3f}, " +
                     f"{np.mean(f1s_test):.3f}+/-{np.std(f1s_test):.3f}, " +
                     f"{np.mean(baseline_f1s):.3f}")

    for x in scores_dict:
        scores_dict[x] = dict(scores_dict[x])
    scores_dict = dict(scores_dict)
    pickle.dump(
        scores_dict,
        open(f"data/f1s_setup={args.setup}_target={args.target}_average={args.average_pie}.pickle", 'wb'))
