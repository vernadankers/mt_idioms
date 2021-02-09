import json
from collections import defaultdict, Counter
import unicodedata
import sys
import argparse
import copy
import nltk
from tqdm import tqdm
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger


nlp = spacy.load("en_core_web_sm")
tagger = SequenceTagger.load('upos-fast')
COLOURS = ['black', 'white', 'red', 'green', 'yellow', 'blue',
           'brown', 'orange', 'pink', 'purple', 'grey']


def get_pos_tags(line, method):
    """
    Returns POS tags for the input line.
    Args:
        line (str): sentence
        method (str): nltk | flair
    Returns
        list of str (POS tags)
    """
    if method == "nltk":
        tokenised_line = nltk.word_tokenize(line)
        _, pos_tags = zip(*nltk.pos_tag(line.split(), tagset="universal"))
    elif method == "flair":
        sentence = Sentence(line, use_tokenizer=False)
        tagger.predict(sentence)
        pos_tags = []
        tokenised_line = []
        for token in sentence.tokens:
            tokenised_line.append(token.text)
            pos_tags.append(token.annotation_layers['pos'][0].value)
    elif method == "spacy":
        line = copy.deepcopy(line)
        doc = nlp(line)
        tokenised_line = [w.text for w in doc]
        pos_tags = [w.pos_ for w in doc]
    else:
        print("I don't know that method")
        exit()
    return tokenised_line, pos_tags


def remove_punctuation(text):
    """
    Removes punctuation from an input string.
    Args:
        text (str): word that potentially contains punctuation
    Returns:
        string with puncutation removed
    """
    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)).startswith('P'))
    return text.translate(tbl)


def preprocess_corpus(corpus, method):
    """
    Load all MAGPIE samples while performing POS tagging.
    Args:
        corpus: list of json object / dicts
        method: nltk | flair for pos tagging
    Returns:
        list of MAGPIE samples in a dict with idioms as keys
        dict mapping idioms to potential POS tags
    """
    idiom_to_pos = defaultdict(list)
    return_samples = defaultdict(list)
    k = 0
    for sample in tqdm(corpus, desc="Iterating over MAGPIE samples"):
        # Exclude samples with confidence lower than 1
        if sample["confidence"] < 1:
            continue

        # Exclude samples that are not fully literal or fully idiomatic
        if sample["label_distribution"]['i'] == 1:
            label = "figurative"
        elif sample["label_distribution"]['l'] == 1:
            label = "literal"
        else:
            continue

        line = sample["context"][2]

        # The dataset provides offsets, using these we can find the words in
        # the sentence that belong to the idiom and collect their pos tags 
        idiom_pos = []
        idiomaticity, extended_idiomaticity = [], []
        char_counter = 0
        tokenised_line, pos_tags = get_pos_tags(line, method)

        for w in line.lower().split():
            tokenised_w, _ = get_pos_tags(w, method)
            if [char_counter, char_counter + len(w)] in sample["offsets"]:
                idiomaticity.append('1')
                extended_idiomaticity.extend(['1'] * len(tokenised_w))
            else:
                idiomaticity.append('0')
                extended_idiomaticity.extend(['0'] * len(tokenised_w))
            char_counter += len(w) + 1

        # if len(extended_idiomaticity) != len(tokenised_line):
        #     unequal_lengths += 1
        else:
            for w, t, a in zip(tokenised_line, pos_tags, extended_idiomaticity):
                if a == '1':
                    idiom_pos.append((w.lower(), t))

        # Collect pos tags separately for the idiom master database
        # assert idiom_pos, f"No POS tags found for {sample}...?"
        idiom_to_pos[sample["idiom"].lower()].extend(idiom_pos)

        # Collect samples to write to file in a tsv format
        return_samples[sample["idiom"].lower()].append((
            line,
            sample["context"],
            label,
            idiomaticity,
            sample["variant_type"]
        ))
    return return_samples, idiom_to_pos


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_tag_method", type=str, default="nltk")
    parser.add_argument("--output_filename", type=str,
                        default="idiom_annotations.tsv")
    args = parser.parse_args()

    # Load the MAGPIE corpus
    magpie = [json.loads(jline) for jline in \
              open("MAGPIE_filtered_split_typebased.jsonl",
                   encoding="utf-8").readlines()]
    subset, idiom_to_pos = preprocess_corpus(magpie, args.pos_tag_method)

    # Extract the keywords based on POS tags
    punctuation_removed = dict()
    with open(args.output_filename, 'w', encoding="utf-8") as f:
        sorted_idioms = sorted(list(idiom_to_pos.keys()))
        for idiom in tqdm(sorted_idioms, desc="Iterating over idioms"):

            # Collect all tags available for one word
            words, tags, keywords = [], [], []
            tags_per_word = defaultdict(list)
            for word, tag in idiom_to_pos[idiom]:
                if word not in punctuation_removed:
                    word_tmp = remove_punctuation(word)
                    punctuation_removed[word] = word_tmp
                    word = word_tmp
                else:
                    word = punctuation_removed[word]
                tags_per_word[word].append(tag)

            maxi = max([len(tags_per_word[word]) for word in tags_per_word])
            # Use the most common tag as the final tag
            for word in tags_per_word:
                if len(tags_per_word[word]) < 0.2 * maxi:
                    continue
                tag = Counter(tags_per_word[word]).most_common(1)[0][0]
                words.append(f"{word}/{tag}")
                tags.append(tag)
                if tag == "NOUN" or tag == "NUM" or word in COLOURS:
                    keywords.append(word)
            
            inclusion = "no" if "NOUN" not in tags else "yes"
            l = f"{idiom}\t{inclusion}\t{' '.join(words)}\t{' '.join(keywords)}"
            f.write(l + '\n')
