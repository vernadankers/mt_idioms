import sys
import random
from collections import defaultdict
import logging
import pickle
sys.path.append('../data/')
from data import extract_sentences
from probing import set_seed
from debias import get_debiasing_projection
logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[])
    parser.add_argument("--attention_layers", type=int, nargs='+', default=[])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+")
    args = parser.parse_args()

    set_seed(1)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.start, args.stop, args.step):
        if (i + 1) % 50 == 0:
            logging.info(f"Sample {i/args.step:.0f} / {(args.stop-args.start)/args.step:.0f}")
        samples = extract_sentences(
            [i], use_tqdm=False, store_hidden_states=args.hidden_layers != [],
            store_attention_query=args.attention_layers != [])
        for s in samples:
            if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                s.label = 1
            elif s.translation_label == "word-by-word" and s.magpie_label == "literal":
                s.label = 0
            else:
                s.label = None

        samples = [s for s in samples if s.label is not None]
        if samples:
            data[i] = samples

    indices = list(data.keys())
    random.shuffle(indices)

    n = int(len(indices)/5)
    fold_1 = indices[:n]
    fold_2 = indices[n:n*2]
    fold_3 = indices[n*2:n*3]
    fold_4 = indices[n*3:n*4]
    fold_5 = indices[n*4:]

    assert 0 not in args.attention_layers, \
        "Cannot intervene om embs with attention queries!"

    percentages, bleus = [], []
    for fold in args.folds:
        if fold == 0:
            train = fold_3 + fold_4 + fold_5
            dev = fold_2
            test = fold_1
        elif fold == 1:
            train = fold_1 + fold_4 + fold_5
            dev = fold_3
            test = fold_2
        elif fold == 2:
            train = fold_1 + fold_2 + fold_5
            dev = fold_4
            test = fold_3
        elif fold == 3:
            train = fold_1 + fold_2 + fold_3
            dev = fold_5
            test = fold_4
        elif fold == 4:
            train = fold_2 + fold_3 + fold_4
            dev = fold_1
            test = fold_5

        train = [s for i in train for s in data[i]]
        dev = [s for i in dev for s in data[i]]

        for layer in range(7):
            if layer in args.hidden_layers:
                P, _, _ = get_debiasing_projection(
                    {"max_iter": 5000, "random_state": 1}, 50, 512,
                    train, dev, layer=layer, attention=False, baseline=args.baseline)
                pickle.dump(
                   P, open(f"projection_matrices/hidden_fold={fold}_layer={layer}_baseline={args.baseline}.pickle", 'wb'))

            if layer in args.attention_layers:
                P, _, _ = get_debiasing_projection(
                    {"max_iter": 5000, "random_state": 1}, 50, 512,
                    train, dev, layer=layer - 1,
                    attention=True, baseline=args.baseline)
                pickle.dump(
                    P, open(f"projection_matrices/attention_fold={fold}_layer={layer}_baseline={args.baseline}.pickle", 'wb'))
