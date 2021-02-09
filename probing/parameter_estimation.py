import random
import copy
import logging
import argparse
import numpy as np
from tqdm import tqdm
from data import extract_encodings, set_seed
from train_probe import main


header = "use_l0\tpos_weight\tlayer\touter_lr\tinner_lr\tn_inner_iter\tmeta_batch_size\tuse_maml\tlambd\tf1_micro\tf1_macro\tremoved_neurons\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, nargs='+', default=0)
    parser.add_argument("--maml", action="store_true")
    parser.add_argument("--l0", action="store_true")
    parser.add_argument("--lambd", type=float, default=0.001)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--key", type=str, default="encodings_idiom")
    parser.add_argument("--keywords_only", action="store_true")
    parser.add_argument("--label_type", type=str, default="magpie")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load all hidden representations of idioms
    set_seed(1)
    data = dict()
    for i in range(0, 1728):
        if (i % 100) == 0:
            logging.info(f"Processed {i} idioms")
        samples = extract_encodings([i], args.label_type, args.key, keywords_only=args.keywords_only)
        if args.key == "encodings_context":
            samples = random.sample(samples, int(0.13 * len(samples)))
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

    fold = 10
    lambd = args.lambd
    use_maml = args.maml
    use_l0 = args.l0

    if not use_maml:
        maml_params = [(0, 0, 4)]

    results = []
    for layer in args.layer:
        for outer_lr in [0.001, 0.0005, 0.00025, 0.0001]:
            for pos_weight in [0.25, 0.5, 0.75]:
                for inner_lr, n_inner_iter, meta_batch_size in maml_params:
                    logging.info(f"SETUP: Layer {layer} | lr {outer_lr} | weight {pos_weight}")
                    # Reset seed
                    set_seed(1)

                    # Separate into training and testing idioms
                    n = int(len(indices) * round(fold / 11, 3))
                    m = int(len(indices) * round((fold + 1) / 11, 3))
                    train_samples = [sample for i in indices[:n] for sample in data[i]] + \
                                    [sample for i in indices[m:] for sample in data[i]]
                    test_samples = [sample for i in indices[n:m] for sample in data[i]]

                    model, f1_micro, f1_macro = \
                        main(train_samples, test_samples,
                             l0=use_l0,
                             layer=layer,
                             outer_lr=outer_lr,
                             epochs=20,
                             pos_weight=pos_weight,
                             inner_lr=inner_lr,
                             n_inner_iter=n_inner_iter,
                             meta_batch_size=meta_batch_size,
                             batch_size=16,
                             maml=use_maml,
                             lambd=lambd)

                    results.append((
                        str(use_l0), str(pos_weight), str(layer),
                        str(outer_lr), str(inner_lr),
                        str(n_inner_iter), str(meta_batch_size),
                        str(use_maml), str(lambd), str(f1_micro),
                        str(f1_macro),
                        str(0 if not use_l0 else 
                            sum([1 for x in model.weights1.mask.forward() if x == 0]))))

                with open(args.output_filename, 'w', encoding="utf-8") as f:
                    f.write(header)
                    for res in results:
                        f.write('\t'.join(res) + '\n')
