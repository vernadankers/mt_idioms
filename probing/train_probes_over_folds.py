import numpy as np
import random
import copy
import argparse
import numpy as np
from tqdm import tqdm
from data import extract_encodings, set_seed
from train_probe import main, test_single
from sklearn.metrics import f1_score

import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load all hidden representations of idioms
    set_seed(1)
    data = dict()
    for i in range(0, 1728):
        if (i % 100) == 0:
            logging.info(f"Processed data from {i} idioms")
        if args.subset == "encodings_idiom":
            samples = extract_encodings([i], "magpie", "encodings_idiom", keywords_only=False)
        elif args.subset == "encodings_context":
            samples = extract_encodings([i], "magpie", "encodings_context", keywords_only=False)
            samples = random.sample(samples, 0.13 * len(samples))
        else:
            samples = extract_encodings([i], "magpie", "encodings_idiom", keywords_only=True)
    
        data[i] = samples

    # Delete empty keys
    data2 = copy.deepcopy(data)
    for x in data:
        if not data[x]:
            del data2[x]
    data = data2

    # Shuffle idioms
    indices = sorted(list(data.keys()))
    random.shuffle(indices)

    mean_per_layer, std_per_layer = [], []
    for layer in range(7):
        logging.info(f"LAYER {layer}")
        # Separate into training and testing idioms
        f1_macros, f1_micros = [], []
        for fold in range(10):
            logging.info(f"FOLD {fold}")
            set_seed(1)
            n = int(len(indices) * round(fold/11, 3))
            m = int(len(indices) * round((fold+1)/11, 3))

            train_samples = [sample for i in indices[:n] for sample in data[i]] + \
                            [sample for i in indices[m:] for sample in data[i]]
            test_samples = [sample for i in indices[n:m] for sample in data[i]]

            model, f1_micro, f1_macro = \
                main(train_samples, test_samples, l0=False, layer=layer, outer_lr=0.00025, epochs=20,
                     pos_weight=0.5, inner_lr=0.5, n_inner_iter=3, meta_batch_size=1,
                     maml=False, lambd=0.001)

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
    logging.info(mean_f1_micro)
    logging.info(mean_f1_macro)
    logging.info(std_f1_micro)
    logging.info(std_f1_macro)
    logging.info(baseline_micro)
    logging.info(baseline_macro)