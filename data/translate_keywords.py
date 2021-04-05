import argparse
import string
import spacy
import inflect
import requests
import torch
from tqdm import tqdm
from transformers import MarianTokenizer

api_key = None #### INSERT API KEY
nlp = spacy.load("en_core_web_sm")
nlp_nl = spacy.load('nl_core_news_sm')
inflect = inflect.engine()


COLOUR_TRANSLATIONS = {
    'black' : ["zwart"],
    'white' : ["wit"],
    'red' : ["rood", "rode"],
    'green' : ["groen"],
    'yellow' : ["geel", "gele"],
    'blue' : ["blauw"],
    'brown' : ["bruin"],
    'orange': ["oranje", "sinaasappel"],
    'pink' : ["roze", "rose"],
    'purple' : ["paars"],
    'grey' : ["grijs", "grijze"]
}


def remove_punctuation(word):
    """Removes punctuation from input string and returns a string."""
    for x in string.punctuation:
        word = word.replace(x, '')
    return word


def to_plural_singular(word):
    """Returns the plural form for singular inputs and vice versa."""
    tokens = nlp(word)
    if not inflect.singular_noun(tokens[0].text):
        return tokens[0]._.inflect('NNS')
    return tokens[0]._.inflect('NN')


def translate(word):
    """Returns a set of possible translations for the input word."""
    translations = []
    if word in COLOUR_TRANSLATIONS:
        return set(COLOUR_TRANSLATIONS[word])

    for word in [word, to_plural_singular(word)]:
        if word is None:
            continue

        r = requests.post(url='https://api.deepl.com/v2/translate',
            data = {'source_lang' : 'EN', 'target_lang' : 'NL',  
                    'auth_key' : "b1661afe-e729-cb33-a631-6200754437e4",
                    'text': word})
        translation = eval(r.text)['translations'][0]['text']
        if word not in translation:
            translations.append(translation)

        batch = tok.prepare_seq2seq_batch(
            src_texts=[f"He sees a {word}"], return_tensors="pt")

        gen = model.generate(
            **batch, num_beams=5, max_length=15, num_return_sequences=5,
            do_sample=False, output_scores=True, return_dict_in_generate=True)
        words = tok.batch_decode(gen["sequences"], skip_special_tokens=True)

        for i, (w, p) in enumerate(zip(words, torch.exp(gen["sequences_scores"]))):
            if (p > 0.5 or i == 0) and "vertalen" not in w and "woord" not in w:
                translation = w.split()[-1].strip().lower()
                if word not in translation:
                    translations.append(translation)
    words = {remove_punctuation(w) for w in translations}
    lemmatized_words = []
    for w in words:
        for w2 in nlp_nl(w):
            lemmatized_words.append(w2.lemma_)
    words = list(words) + list(lemmatized_words)
    return set(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str,
                        default="idiom_annotations.tsv")
    parser.add_argument("--output_tsv", type=str,
                        default="idiom_annotations_translated.tsv")
    args = parser.parse_args()

    mname = f'Helsinki-NLP/opus-mt-en-nl'
    model = MarianMTModel.from_pretrained(mname)
    tok = MarianTokenizer.from_pretrained(mname)

    with open(args.input_tsv, encoding="utf-8") as f_in, \
         open(args.output_tsv, 'w', encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Iterating over idioms"):
            idiom, inclusion, words_tags, keywords = line.split("\t")

            if inclusion == "no":
                continue

            all_translations = []
            for keyword in keywords.strip().split():
                all_translations.append(';'.join(translate(keyword)))

            l = f"{idiom}\t{inclusion}\t{words_tags}\t{keywords.strip()}"
            f_out.write(l + f"\t{' '.join(all_translations)}" + '\n')
