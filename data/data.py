import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from torch import FloatTensor as FT, LongTensor as LT
from classifier import Classifier
import pickle5 as pickle


classifier = Classifier(
    "../data/idiom_keywords_translated.tsv")


class Sentence():
    def __init__(self, sentence, tokenised_sentence, hidden_states_enc, attention,
                 attention_query, cross_attention, translation_label,
                 magpie_label, variant, idiom,
                 annotation, tokenised_annotation, pos_tags,
                 non_tokenised_pos_tags, prd=None):
        self.sentence = sentence
        self.translation = prd
        self.tokenised_sentence = tokenised_sentence
        self.tokenised_annotation = tokenised_annotation
        src_len = len(self.tokenised_annotation) + 1

        if hidden_states_enc is not None:
            self.hidden_states = FT(hidden_states_enc)

        if attention_query is not None:
            self.attention_query = FT(attention_query)

        if attention is not None:
            # Identify which indices were masked in the attention,
            # is only used in the influence setup for SVCCA similarities
            self.masked_indices = []
            for l in range(6):
                summed = np.sum(np.sum(attention[l], axis=0), axis=0)[:src_len]
                indices = [1 if x == 0 else 0 for i, x in enumerate(summed)]
                self.masked_indices.append(indices)
            self.attention = FT(attention)[:, :, :src_len, :src_len]

        if cross_attention is not None:
            self.cross_attention = FT(cross_attention)[:, :, :len(self.translation), :src_len]

        self.translation_label = translation_label
        self.magpie_label = magpie_label
        self.variant = variant
        self.idiom = idiom
        self.annotation = annotation
        self.pos_tags = pos_tags
        self.non_tokenised_pos_tags = non_tokenised_pos_tags

    def index_select(self, target_label, tags=[], no_subtokens=True,
                     neighbours_only=False, neighbourhood=10,
                     context_context=False, layer=None):
        """Collect all indices that adhere to certain conditions."""
        if isinstance(target_label, int):
            target_label = [target_label]
        neighbours = []

        # Collect neighbours of idiom, unless context_context
        # in that case take the neighbours from the masked indices
        ann = self.tokenised_annotation if not context_context else self.masked_indices[layer]

        # Now collect the neighbours
        for i, l in enumerate(ann):
            if l != 1 and 1 in ann[max(0, i-neighbourhood):i + neighbourhood + 1]:
                neighbours.append(i)

        # Collect all sentences that adhere to the conditions
        indices = []
        for i, word in enumerate(self.tokenised_sentence.split()):
            # They should have the given label type (0 or 1)
            is_label = self.tokenised_annotation[i] in target_label
            # They shouldn't be subtokens if asked for
            not_subtoken = u'▁' in word if no_subtokens else True
            # Their POS tag should be in the provided list of tags
            correct_tag = self.pos_tags[i] in tags if tags else True
            # They should be a neighbour to token annotated with a 1
            is_neighbour = i in neighbours if neighbours_only else True

            if is_label and not_subtoken and correct_tag and is_neighbour:
                indices.append(i)
        return indices


def extract_sentences(indices,
                      use_tqdm=False,
                      data_folder="../data/magpie", 
                      store_hidden_states=False,
                      store_attention=False,
                      store_cross_attention=False,
                      store_attention_query=False,
                      influence_setup=False,
                      get_verb_idioms=False):
    sentences = []

    for i in tqdm(indices, disable=not use_tqdm):
        per_idiom = []

        # Only open the pickled data needed, to avoid exploding memory usage
        if store_hidden_states:
            hidden_states_enc = pickle.load(open(
                f"{data_folder}/hidden_states_enc/{i}_pred_hidden_states_enc.pickle", 'rb'))

        if store_attention_query:
            attention_queries = pickle.load(open(
                f"{data_folder}/query_attention/{i}_pred_query_attention.pickle", 'rb'))

        if store_attention:
            attention = pickle.load(open(
                f"{data_folder}/attention/{i}_pred_attention.pickle", 'rb'))

        if store_cross_attention:
            tokenised_prds = pickle.load(open(
                f"{data_folder}/tokenised_prds/{i}_pred_prds.pickle", 'rb'))
            cross_attention = pickle.load(open(
                f"{data_folder}/cross_attention/{i}_pred_cross_attention.pickle", 'rb'))

        if influence_setup:
            data = pickle.load(open(f"{data_folder}/{i}_pred.pickle", 'rb'))
            hidden_states_enc = data["hidden_states"]
            attention = data["attention"]
            store_hidden_states = True
            store_attention = True

        # Open all precomputed data info, such as tokenised annotations
        srcs = []
        data_info = defaultdict(lambda: dict)
        for x in open(f"../data/magpie/inputs/{i}.tsv", encoding="utf-8"):
            sentence, annotation, idiom, label, variant, tags, \
                tok_sent, tok_annotation, tok_tags = x.split("\t")
            srcs.append(sentence)
            data_info[sentence] = {
                "annotation": annotation,
                "idiom": idiom,
                "label": label,
                "variant": variant,
                "tags": tags.strip().split(),
                "tok_sent": tok_sent,
                "tok_annotation": [int(l) for l in tok_annotation.split()],
                "tok_tags": tok_tags
            }

        # If this idiom is a verb idiom, continue, unless get_verb_idioms is on
        idiom = list(data_info.values())[0]["idiom"]
        if (not get_verb_idioms and not classifier.contains(idiom)) or \
           (get_verb_idioms and classifier.contains(idiom)):
            continue

        # Collect the detokenised predicted translations by the model
        prds = open(f"../data/magpie/prds/{i}_pred.txt", encoding="utf-8").readlines()
        to_prds = {x.strip() : y.strip() for x, y in zip(srcs, prds)}

        for sent in data_info:
            # We only generated translations up to length 512, so remove long items
            if len(data_info[sent]["tok_annotation"]) >= 511:
                continue

            translation_label = classifier(idiom, to_prds[sent])

            # Exclude copies, only keep other samples if get_verb_idioms is True
            if translation_label in ["copied"]:
                continue
            if not get_verb_idioms and translation_label in ["none"]:
                continue

            # Create a Sentence object containing all info asked for
            per_idiom.append(Sentence(
                sentence=sent,
                tokenised_sentence=data_info[sent]["tok_sent"],
                hidden_states_enc=None if not store_hidden_states else hidden_states_enc[sent],
                attention=None if not store_attention else attention[sent],
                attention_query=None if not store_attention_query else attention_queries[sent],
                cross_attention=None if not store_cross_attention else cross_attention[sent],
                translation_label=translation_label,
                magpie_label=data_info[sent]["label"],
                variant=data_info[sent]["variant"],
                idiom=idiom,
                annotation=[int(l) for l in data_info[sent]["annotation"].split()],
                tokenised_annotation=data_info[sent]["tok_annotation"],
                pos_tags=data_info[sent]["tok_tags"].strip().split(),
                non_tokenised_pos_tags=data_info[sent]["tags"],
                prd=to_prds[sent] if not store_cross_attention else tokenised_prds[sent]))
        sentences.extend(per_idiom)
    return sentences
