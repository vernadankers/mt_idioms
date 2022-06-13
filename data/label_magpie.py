from collections import defaultdict, Counter
import pickle
import os
import tqdm
from classifier import Classifier
import argparse
import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main(language, use_spacy):
    set_seed(1)
    classifier = Classifier(
        f'keywords/idiom_keywords_translated_{language}.tsv',
        language=language,
        use_spacy=use_spacy)

    unique_idioms = []
    trace = defaultdict(lambda: defaultdict(list))
    for i in tqdm.tqdm(range(0, 1727)):
        if not os.path.exists(f'magpie/{language}/prds/{i}_pred.txt'):
            continue
        with open(f'magpie/inputs/{i}.tsv', encoding='utf-8') as f_src, \
                open(f'magpie/{language}/prds/{i}_pred.txt', encoding='utf-8') as f_prd:
            for src, prd in zip(f_src, f_prd):
                # Load and clean data
                src, _, idiom, label = src.split('\t')[:4]
                idiom = idiom.strip()
                src = src.strip()
                prd = prd.strip()

                # Classify translation
                prd_label = classifier(idiom, prd)
                if prd_label == 'none':
                    continue

                # Record per label subset
                trace[(label, prd_label)][idiom].append((src, prd))
                trace[label][idiom].append((src, prd))

                if idiom not in unique_idioms:
                    if "guess" in idiom:
                        print(idiom, src, prd)
                        print("your guess is as good as mine" == idiom)
                    unique_idioms.append(idiom)

    # Save classified data to file
    for x in trace:
        trace[x] = dict(trace[x])
    trace = dict(trace)
    pickle.dump(trace, open(f'classified_data_{language}.pickle', 'wb'))
    counter = Counter({x: sum([len(trace[x][y])
                      for y in trace[x]]) for x in trace})


    # Report statistics to the user
    print(f"#idioms: {len(unique_idioms)}")
    total = counter['figurative'] + counter['literal']
    # print(f"{total} samples total.")
    total_fig = counter['figurative']
    total_lit = counter['literal']
    print(f"Figurative: {counter['figurative'] / total:.3f}")
    print(f"Literal: {counter['literal'] / total:.3f}")

    fig_par = counter[('figurative', 'paraphrase')] / total_fig
    fig_wfw = counter[('figurative', 'word-by-word')] / total_fig
    lit_par = counter[('literal', 'paraphrase')] / total_lit
    lit_wfw = counter[('literal', 'word-by-word')] / total_lit
    print(f"{fig_par:.2f} & {fig_wfw:.2f} \\\\")
    print(f"{lit_par:.2f} & {lit_wfw:.2f} \\\\")

    if language == "nl":
        set_seed(1)
        random.shuffle(unique_idioms)
        n = int(len(unique_idioms)/5)
        fold_1 = unique_idioms[:n]
        fold_2 = unique_idioms[n:n*2]
        fold_3 = unique_idioms[n*2:n*3]
        fold_4 = unique_idioms[n*3:n*4]
        fold_5 = unique_idioms[n*4:]
        print(fold_1[0], fold_2[0], fold_3[0], fold_4[0], fold_5[0])
        assert not set(fold_1).intersection(set(fold_2))
        pickle.dump({0: fold_1, 1: fold_2, 2: fold_3, 3: fold_4, 4: fold_5}, open("folds.pickle", 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--spacy", action='store_true')
    args = parser.parse_args()
    main(args.language, args.spacy)
