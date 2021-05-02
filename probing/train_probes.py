import sys
import random
import argparse
import logging
import numpy as np
from sklearn.metrics import f1_score
import torch
from probe import Trainer
sys.path.append('../data/')
from data import extract_sentences


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_type", type=str, required=True)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--m", type=int, default=1728)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--average_over_pie", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.n, args.m, args.k):
        samples = extract_sentences([i], use_tqdm=False, store_hidden_states=True)

        for s in samples:
            if args.label_type == "magpie":
                s.label = 1 if s.magpie_label == "figurative" else 0
            elif args.label_type == "both":
                if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                    s.label = 1
                elif s.translation_label == "word-by-word" and s.magpie_label == "literal":
                    s.label = 0
                else:
                    s.label = None
            else:
                s.label = 1 if s.translation_label == "paraphrase" else \
                          (0 if s.translation_label == "word-by-word" else None)

        samples = [s for s in samples if s.label is not None]
        if samples:
            data[i] = samples

    # Shuffle idioms
    set_seed(1)
    indices = sorted(list(data.keys()))
    random.shuffle(indices)

    mean_per_layer, std_per_layer = [], []
    for layer in range(7):
        logging.info(f"LAYER {layer}")

        # Separate into training and testing idioms
        f1_macros, f1_micros = [], []
        for seed in range(5):
            set_seed(seed)

            for fold in range(5):
                # Initialise the probing classifier and training functionalities
                trainer = Trainer(
                    lr=0.0005,
                    epochs=5,
                    layer=layer,
                    batch_size=16,
                    average_over_pie=args.average_over_pie)
                trainer.init_model()
                logging.info(f"FOLD {fold}")
                n = int(len(indices) * (fold/5))
                m = int(len(indices) * ((fold+1)/5))
                train_samples = [sample for i in indices[:n] for sample in data[i]] + \
                                [sample for i in indices[m:] for sample in data[i]]
                test_samples = [sample for i in indices[n:m] for sample in data[i]]
                f1_micro, f1_macro = trainer.train(train_samples, test_samples)
                f1_macros.append(f1_macro)
                f1_micros.append(f1_micro)
        mean_per_layer.append((np.mean(f1_micros), np.mean(f1_macros)))
        std_per_layer.append((np.std(f1_micros), np.std(f1_macros)))

    labels = [sample.label for i in indices for sample in data[i]]
    baseline = [random.choice(labels) for _ in labels]
    baseline_micro = f1_score(labels, baseline, average="micro")
    baseline_macro = f1_score(labels, baseline, average="macro")
    mean_f1_micro, mean_f1_macro = zip(*mean_per_layer)
    std_f1_micro, std_f1_macro = zip(*std_per_layer)

    logging.info(f"f1_micro = {[round(x, 3) for x in mean_f1_micro]}")
    logging.info(f"f1_macro = {[round(x, 3) for x in mean_f1_macro]}")
    logging.info(f"f1_micro_std = {[round(x, 3) for x in std_f1_micro]}")
    logging.info(f"f1_macro_std = {[round(x, 3) for x in std_f1_macro]}")
    logging.info(f"f1_micro_baseline = {[round(baseline_micro, 3) for _ in mean_f1_micro]}")
    logging.info(f"f1_macro_baseline = {[round(baseline_macro, 3) for _ in mean_f1_macro]}")
