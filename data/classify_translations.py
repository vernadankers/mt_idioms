from collections import defaultdict
import argparse
import pickle


class Classifier():
    def __init__(self, idiom_file):
        self.idioms = []
        self.nl_keywords = dict()
        self.en_keywords = dict()
        with open(idiom_file, encoding="utf-8") as f_annot:
            f_annot.readline()
            for line in f_annot:
                idiom, contains_noun, translation, en_keywords, nl_keywords, pos_tags = \
                    line.strip().split("\t")[:6]
                self.idioms.append(idiom)
                if contains_noun == "yes":
                    self.nl_keywords[idiom] = nl_keywords.split()
                    self.en_keywords[idiom] = en_keywords.split()

    def __call__(self, idiom, src, prd):
        return self.classify(idiom, src, prd)

    def classify(self, idiom, src, prd):
        if type(idiom) == int:
            idiom = self.idioms[idiom]
        idiom = idiom.lower().strip()
        if not idiom in self.nl_keywords:
            return "none"
        src = src.lower().strip()
        prd = prd.lower().strip()

        keywords = self.en_keywords[idiom]
        keywords_present = []
        for keyword in keywords:
            keywords_present.append(any([w in prd for w in keyword.split(';')]))
        if all(keywords_present):
            return "copied"

        keywords = self.nl_keywords[idiom]
        keywords_present = []
        for keyword in keywords:
            keywords_present.append(any([w in prd for w in keyword.split(';')]))
        if any(keywords_present):
            return "word-by-word"
        return "paraphrase"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--prd", type=str, required=True)
    parser.add_argument("--annotation_file", type=str,
        default="../data/magpie-corpus/idiom_annotations.tsv")
    args = parser.parse_args()

    classifier = Classifier(args.annotation_file)

    trace = defaultdict(lambda: defaultdict(list))
    with open(args.src, encoding="utf-8") as f_src, \
         open(args.prd, encoding="utf-8") as f_prd:
        for src, prd in zip(f_src, f_prd):
            src, annotations, idiom, label, variant = src.split("\t")
            idiom = idiom.strip()
            src = src.strip()
            prd = prd.strip()
            prd_label = classifier(idiom, src, prd)
    #         trace[(label, prd_label)][idiom].append((src, prd))

    # for x in trace:
    #     trace[x] = dict(trace[x])
    # trace = dict(trace)
