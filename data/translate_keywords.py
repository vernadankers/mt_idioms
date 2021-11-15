import argparse
import string
import spacy
import inflect
import requests
import torch
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
import six
from google.cloud import translate_v2
import pyinflect
import spacy_stanza

# def get_translation(word, language):
#     r = requests.post(url='https://api-free.deepl.com/v2/translate',
#         data = {'source_lang' : 'EN', 'target_lang' : language.upper(),
#                 'auth_key' : "bd21201b-89fd-6f66-2799-90b09bebc05e:fx",
#                 'text': word})
#     translation = eval(r.text)['translations'][0]['text']
#     return translation

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mt-idioms-e989ed52dafb.json"



def get_translation(text, language):
    translate_client = translate_v2.Client()

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=language, source_language='en')
    return result["translatedText"]


def remove_punctuation(word):
    """Removes punctuation from input string and returns a string."""
    for x in ['.', '?', '!']:
        word = word.replace(x, '')
    return word


def to_plural_singular(word):
    """Returns the plural form for singular inputs and vice versa."""
    tokens = nlp(word)
    if not inflect.singular_noun(tokens[0].text):
        return tokens[0]._.inflect('NNS')
    return tokens[0]._.inflect('NN')


def translate(word, language, marian, google_translate, spacy):
    """Returns a set of possible translations for the input word."""

    ## DeepL translations
    deepl_translation = ""

    # r = requests.post(url='https://api.deepl.com/v2/translate',
    #     data = {'source_lang' : 'EN', 'target_lang' : 'NL',
    #             'auth_key' : "b1661afe-e729-cb33-a631-6200754437e4",
    #             'text': word})
    # translation = eval(r.text)['translations'][0]['text']
    #print(get_translation(f"Translate the following noun: {word}.", language))
    if google_translate:
        deepl_translation = get_translation(word, language).strip()

        deepl_translation = remove_punctuation(deepl_translation).split()[-1]
        if spacy:
            for w2 in nlp_nl(deepl_translation):
                lemma = w2.lemma_
                break
        else:
            lemma = deepl_translation
        deepl_translation = {deepl_translation, lemma}

    ## Marian-MT translations
    marian_translations = []
    if marian:
        marian_words = set()
        for word in [word, to_plural_singular(word)]:
            batch = tok.prepare_seq2seq_batch(
                src_texts=[f"Translate this noun: {word}.", f"This is a {word}."], return_tensors="pt") # [f"Translate this noun: {word}.", f"This is a {word}."]
            print(batch)
            for x in batch:
                if torch.cuda.is_available():
                    batch[x] = batch[x].cuda()

            gen = model.generate(
                **batch, num_beams=5, max_length=15, num_return_sequences=5,
                do_sample=False, output_scores=True, return_dict_in_generate=True)
            words = tok.batch_decode(
                gen["sequences"], skip_special_tokens=True)

            for i, (w, p) in enumerate(zip(words, torch.exp(gen["sequences_scores"]))):
                translation = w.split()[-1][:-1].strip()

                if translation not in marian_words:
                    if spacy:
                        for w2 in nlp_nl(translation):
                            lemma = w2.lemma_
                            translation = w2.text
                    else:
                        lemma = translation
                    lemma = remove_punctuation(lemma)
                    translation = remove_punctuation(translation)
                    if translation not in marian_words:
                        marian_translations.append(
                            translation)
                        marian_translations.append(lemma)
                        marian_words.add(translation)

    return deepl_translation, set(marian_translations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str,
                        default="idiom_annotations.tsv")
    parser.add_argument("--output_tsv", type=str,
                        default="idiom_annotations_translated.tsv")
    parser.add_argument("--language", type=str, default="nl")
    parser.add_argument("--marian", action="store_true")
    parser.add_argument("--google_translate", action="store_true")
    parser.add_argument("--spacy", action="store_true")
    args = parser.parse_args()

    mname = f'Helsinki-NLP/opus-mt-en-{args.language}'
    model = MarianMTModel.from_pretrained(mname)
    if torch.cuda.is_available():
        model = model.cuda()
    tok = MarianTokenizer.from_pretrained(mname)

    api_key = None  # INSERT API KEY
    nlp = spacy.load("en_core_web_sm")
    if args.spacy:
        if args.language == "sv":
            nlp_nl = spacy_stanza.load_pipeline(args.language)
        else:
            nlp_nl = spacy.load(f'{args.language}_core_news_sm')
    else:
        nlp_nl = None
    inflect = inflect.engine()

    print("Yup")

    with open(args.input_tsv, encoding="utf-8") as f_in, \
            open(args.output_tsv, 'w', encoding="utf-8") as f_out:
        for i, line in tqdm(enumerate(f_in)):
            idiom, inclusion, words_tags, keywords = line.split("\t")

            if inclusion == "no":
                continue

            all_deepl_translations = []
            all_marian_translations = []
            for keyword in keywords.strip().split():
                deepl_translation, marian_translation = translate(
                    keyword, args.language, args.marian, args.google_translate, args.spacy)
                all_deepl_translations.append(';'.join(deepl_translation))
                all_marian_translations.append(';'.join(marian_translation))

            l = f"{idiom}\t{inclusion}\t{words_tags}\t{keywords.strip()}\t{' '.join(all_deepl_translations)}"
            f_out.write(l + f"\t{' '.join(all_marian_translations)}" + '\n')
