import sys
sys.path.append('../data/')
import torch
import gc
import pickle
import logging
import random
from data import extract_sentences
from classifier import Classifier
from transformers import MarianTokenizer
from probing import set_seed
from debias import get_debiasing_projection, \
    get_projection_to_intersection_of_nullspaces
logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--language", type=str, default="nl")
    args = parser.parse_args()

    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{args.language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{args.language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)

    set_seed(1)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.start, args.stop, args.step):
        if (i + 1) % 50 == 0:
            logging.info(
                f"Sample {i/args.step:.0f} / {(args.stop-args.start)/args.step:.0f}")
        samples = extract_sentences(
            [i], classifier, tokenizer, use_tqdm=False, store_hidden_states=True,
            data_folder=f"../data/magpie/{args.language}")
        for s in samples:
            if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                s.label = 1
            elif s.translation_label == "word-by-word" and s.magpie_label == "figurative":
                s.label = 0
            else:
                s.label = None
        if samples:
            idiom = samples[0].idiom
            samples = [s for s in samples if s.label is not None]
            data[idiom] = samples

    folds = pickle.load(open("../data/folds.pickle", 'rb'))
    fold_1 = [s for i in folds[0] for s in data[i]]
    fold_2 = [s for i in folds[1] for s in data[i]]
    fold_3 = [s for i in folds[2] for s in data[i]]
    fold_4 = [s for i in folds[3] for s in data[i]]
    fold_5 = [s for i in folds[4] for s in data[i]]
    del data
    gc.collect()

    for fold in args.folds:
        # Separate into folds
        if fold == 0:
            train = fold_3 + fold_4 + fold_5
            dev = fold_2
        elif fold == 1:
            train = fold_1 + fold_4 + fold_5
            dev = fold_3
        elif fold == 2:
            train = fold_1 + fold_2 + fold_5
            dev = fold_4
        elif fold == 3:
            train = fold_1 + fold_2 + fold_3
            dev = fold_5
        elif fold == 4:
            train = fold_2 + fold_3 + fold_4
            dev = fold_1

        train1 = [x for x in train if x.label == 1]
        train0 = [x for x in train if x.label == 0]

        # Equalise the sizes of the 0 and 1 class. The 0 class is larger,
        # so prune that one.
        random.shuffle(train0)
        train = train0[:len(train1)] + train1
        random.shuffle(train)

        for layer in range(7):
            # Collect projection matrices by training on hidden states
            _, rowspace_projections, _ = get_debiasing_projection(
                {"max_iter": 2500, "random_state": 1}, 50, 512,
                train, dev, layer=layer, baseline=args.baseline)

            P = torch.FloatTensor(get_projection_to_intersection_of_nullspaces(
                rowspace_projections, 512))
            pickle.dump(
                P, open(f"projection_matrices/{args.language}_hidden_fold={fold}_layer=" +
                        f"{layer}_baseline={args.baseline}.pickle", 'wb'))
