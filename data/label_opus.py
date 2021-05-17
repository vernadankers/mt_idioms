import os
from collections import defaultdict, Counter
from classifier import Classifier
import sacrebleu
import numpy as np
import nltk
import tqdm


def main():
    idioms = open("idiom_keywords_translated.tsv", encoding="utf-8").readlines()
    idioms = [x.split("\t")[0].strip() for x in idioms]


    classifier = Classifier("idiom_keywords_translated.tsv")
    counter = Counter()
    data = defaultdict(list)

    tgt_wbw, prd_wbw = [], []
    tgt_par, prd_par = [], []

    occs = []

    x = 0
    for i in tqdm.tqdm(range(0, 1727)):
        if not os.path.exists(f"opus/{i}_pred.txt"):
            continue
        with open(f"opus/{i}.en", encoding="utf-8") as f_src, \
             open(f"opus/{i}.nl", encoding="utf-8") as f_tgt, \
             open(f"opus/{i}_pred.txt", encoding="utf-8") as f_prd:
            f_src = f_src.readlines()
            f_tgt = f_tgt.readlines()
            f_prd = f_prd.readlines()
            occs.append(len(f_src))

            if not (len(f_src) == len(f_tgt) and len(f_src) == len(f_prd)):
                continue

            for src, tgt, prd in zip(f_src, f_tgt, f_prd):
                if not src or not tgt:
                    continue
                prd = prd.split("\t")[0].strip()
                idiom = idioms[i]

                if not classifier.contains(idiom):
                    continue

                label = classifier(idiom, tgt)
                prd_label = classifier(idiom, prd)

                if label == "none" or prd_label == "none":
                    continue

                tgt = " ".join(nltk.word_tokenize(tgt.lower()))
                prd = " ".join(nltk.word_tokenize(prd.lower()))
                prd_par.append(prd)
                data[(label, prd_label)].append((prd, tgt))
                counter[(label, prd_label)] += 1
                counter[label] += 1


    denominator = (counter['paraphrase'] + counter['word-by-word'] + counter['copied'])
    print(f"% of paraphrases {counter['paraphrase'] / denominator:.3f}")
    print(f"% of word for word {counter['word-by-word'] / denominator:.3f}")
    print(f"% of copied instances {counter['copied'] / denominator:.3f}")

    print(f"Paraphrase: % of paraphrased {counter[('paraphrase', 'paraphrase')] / counter['paraphrase']:.3f}")
    print(f"Paraphrase: % of word for word {counter[('paraphrase', 'word-by-word')] / counter['paraphrase']:.3f}")
    print(f"Paraphrase: % of copies {counter[('paraphrase', 'copied')] / counter['paraphrase']:.3f}")

    print(f"Word for word: % of paraphrases {counter[('word-by-word', 'paraphrase')] / counter['word-by-word']:.3f}")
    print(f"Word for word: % of word for word {counter[('word-by-word', 'word-by-word')] / counter['word-by-word']:.3f}")
    print(f"Word for word: % of copies {counter[('word-by-word', 'copied')] / counter['word-by-word']:.3f}")

    print(f"Copy: % of paraphrases {counter[('copied', 'paraphrase')] / counter['copied']:.3f}")
    print(f"Copy: % of word for word {counter[('copied', 'word-by-word')] / counter['copied']:.3f}")
    print(f"Copy: % of copies {counter[('copied', 'copied')] / counter['copied']:.3f}")

    for label1 in ["paraphrase", "word-by-word", "copied"]:
        for label2 in ["paraphrase", "word-by-word", "copied"]:
            print(label1, label2)
            prds, tgts = zip(*data[(label1, label2)])
            bleu = sacrebleu.corpus_bleu(prds, [tgts])
            print(f"{label1} - {label2} - BLEU score: {bleu.score:.1f}")


if __name__ == '__main__':
    main()
