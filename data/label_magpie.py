from collections import defaultdict, Counter
import pickle
import os
from classifier import Classifier


def main():
    classifier = Classifier('idiom_keywords_translated.tsv')

    trace = defaultdict(lambda: defaultdict(list))
    for i in range(1727):
        if not os.path.exists(f'magpie/prds/{i}_pred.txt'):
            continue
        with open(f'magpie/inputs/{i}.tsv', encoding='utf-8') as f_src, \
            open(f'magpie/prds/{i}_pred.txt', encoding='utf-8') as f_prd:
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

    # Save classified data to file
    for x in trace:
        trace[x] = dict(trace[x])
    trace = dict(trace)
    pickle.dump(trace, open('classified_data.pickle', 'wb'))
    counter = Counter({x : sum([len(trace[x][y]) for y in trace[x]]) for x in trace})

    # Report statistics to the user
    print(f"#idioms: {len(set(trace['literal'].keys()).union(set(trace['figurative'].keys())))}")
    total = counter['figurative'] + counter['literal']
    print(f"{total} samples total.")
    total_fig = counter['figurative']
    total_lit = counter['literal']
    print(f"Figurative: {counter['figurative'] / total:.3f}")
    print(f"Literal: {counter['literal'] / total:.3f}")
    print(f"Fig-par: {counter[('figurative', 'paraphrase')] / total_fig:.3f}")
    print(f"Fig-wfw: {counter[('figurative', 'word-by-word')] / total_fig:.3f}")
    print(f"Fig-cop: {counter[('figurative', 'copied')] / total_fig:.3f}")
    print(f"Lit-par: {counter[('literal', 'paraphrase')] / total_lit:.3f}")
    print(f"Lit-wfw: {counter[('literal', 'word-by-word')] / total_lit:.3f}")
    print(f"Lit-cop: {counter[('literal', 'copied')] / total_lit:.3f}")


if __name__ == '__main__':
    main()
