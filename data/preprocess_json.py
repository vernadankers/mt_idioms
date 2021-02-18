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

    for sample in tqdm(corpus[:5000], desc="Iterating over MAGPIE samples"):
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

        if len(tokenised_line) == len(pos_tags) == len(extended_idiomaticity):
            for w, t, a in zip(tokenised_line, pos_tags, extended_idiomaticity):
                if a == '1':
                    idiom_pos.append((w.lower(), t))

            # Collect pos tags separately for the idiom master database
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


def corpus_to_files(samples):
    """Create a tsv per idiom."""
    idioms = sorted(samples.keys())
    for i, idiom in tqdm(enumerate(idioms), "Saving samples in .tsv files per idiom"):
        with open(f"per_idiom2/{i}.tsv", 'w', encoding="utf-8") as f_out:
            for sample in samples[idiom]:
                sentence, _, label, annotation, variant_type = sample
                tags = tag(sentence)
                assert len(tags) == len(sentence.split())
                f_out.write(f"{sentence}\t{' '.join(annotation)}\t{idiom}\t" + \
                            f"{label}\t{variant_type}\t{' '.join(tags)}\n")


def tag(sentence):
    """POS tag a sentence with SpaCy without changing the number of tokens."""
    tags = [w.pos_ for w in nlp(sentence)]
    tags_pruned = []

    for w in sentence.split():
        tags2 = [tags.pop(0) for w2 in nlp(w)]
        if set(tags2) != {"PUNCT"} and "PUNCT" in tags2:
            tags2 = list(filter(("PUNCT").__ne__, tags2))
        tags_pruned.append(tags2[0])
    return tags_pruned



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_tag_method", type=str, default="nltk")
    parser.add_argument("--output_filename", type=str,
                        default="idiom_annotations.tsv")
    parser.add_argument("--samples_to_file", action="store_true")
    parser.add_argument("--frequency_threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load the MAGPIE corpus
    magpie = [json.loads(jline) for jline in \
              open("MAGPIE_filtered_split_typebased.jsonl",
                   encoding="utf-8").readlines()]
    samples, idiom_to_pos = preprocess_corpus(magpie, args.pos_tag_method)
    if args.samples_to_file:
        corpus_to_files(samples)

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
                if len(tags_per_word[word]) < (args.frequency_threshold * maxi):
                    continue
                tag = Counter(tags_per_word[word]).most_common(1)[0][0]
                words.append(f"{word}/{tag}")
                tags.append(tag)
                if tag == "NOUN" or tag == "NUM" or word in COLOURS:
                    keywords.append(word)
            
            inclusion = "no" if "NOUN" not in tags else "yes"
            l = f"{idiom}\t{inclusion}\t{' '.join(words)}\t{' '.join(keywords)}"
            f.write(l + '\n')
