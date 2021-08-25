import pickle
import tqdm
from collections import defaultdict
from torch import FloatTensor as FT


class Sentence():
    def __init__(self, sentence, tokenised_sentence, translation, attention,
                 cross_attention, tokenised_annotation, pos_tags):
        self.sentence = sentence
        self.tokenised_sentence = tokenised_sentence
        self.tokenised_annotation = tokenised_annotation
        self.pos_tags = pos_tags

        src_len = len(self.tokenised_annotation) + 1
        prd_len = len(translation)
        self.translation = translation
        self.attention = FT(attention)[:, :, :src_len, :src_len]
        self.cross_attention = FT(cross_attention)[:, :, :prd_len, :src_len]

    def not_subtoken(self, index):
        return u'▁' in self.tokenised_sentence.split()[index]

    def index_select(self, target_label, tags=[], no_subtokens=True,
                     neighbours_only=False, neighbourhood=5):
        if isinstance(target_label, int):
            target_label = [target_label]
        neighbours_annotation = self.tokenised_annotation
        neighbours = [i for i, l in enumerate(neighbours_annotation)
                      if 1 in neighbours_annotation[max(0, i-neighbourhood):i+neighbourhood+1]
                      and neighbours_annotation[i] != 1]
        return [i for i, (w, l, t) in
                enumerate(zip(self.tokenised_sentence.split(),
                              self.tokenised_annotation, self.pos_tags))
                if l in target_label
                and (u'▁' in w if no_subtokens else True)
                and (t in tags if tags else True)
                and (i in neighbours if neighbours_only else True)]


def extract_sentences(lang, disable_tqdm=False):
    samples = []

    for dataset in ["os18", "wmt19"]:
        for start, stop in [(0, 1000), (1000, 2000), (2000, 3000)]:
            pickled_data = pickle.load(
                open(f"data/{dataset}_{start}-{stop}_{lang}_pred.pickle", 'rb'))

            data_info = defaultdict(lambda: dict)
            tsv_lines = open(
                f"data/{dataset}_{start}-{stop}_{lang}.tsv", encoding="utf-8").readlines()
            for line in tqdm.tqdm(tsv_lines, disable=disable_tqdm):
                sentence, _, homograph, _, tok_sent, tok_annotation, tok_tags = line.split("\t")[
                                                                                           :-1]
                data_info[sentence] = {
                    "homograph": homograph,
                    "tok_sent": tok_sent,
                    "tok_annotation": tok_annotation,
                    "tok_tags": tok_tags
                }

            for sent in data_info:
                samples.append(Sentence(
                    sent,
                    data_info[sent]["tok_sent"],
                    pickled_data["prds"][sent],
                    pickled_data["attention"][sent],
                    pickled_data["cross_attention"][sent],
                    [int(l)
                     for l in data_info[sent]["tok_annotation"].split()],
                    data_info[sent]["tok_tags"].strip().split()))
    return samples
