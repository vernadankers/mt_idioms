import spacy
import inflect
import pyinflect
import dutch_words
import nltk
import stanza
import spacy_stanza

inflect = inflect.engine()

class Classifier():
    """Simple classifier to label translations according to whether they
    contain the translation of a keywor (or not)."""

    def __init__(self, idiom_file: str, exclude_deepl: bool = False, use_spacy: bool = False, language: str = "nl"):
        self.idioms = []
        self.src_keywords = dict()
        self.tgt_keywords = dict()

        self.use_spacy = use_spacy
        self.nlp_en = spacy.load("en_core_web_sm")
        if use_spacy:
            self.nlp_tgt = spacy.load(f"{language}_core_news_sm")

        # Open the idioms that contain nouns from file
        with open(idiom_file, encoding="utf-8") as f_annot:
            for line in f_annot:

                if len(line.split("\t")) == 5:
                    if line.strip().split("\t")[1] == "no":
                        continue
                    idiom, inclusion, _, src_keywords, marian_keywords = \
                        line.strip().split("\t")
                    deepl_keywords = ""
                else:
                    idiom, inclusion, _, src_keywords, deepl_keywords, marian_keywords = \
                        line.strip().split("\t")
                if inclusion == "yes":
                    self.idioms.append(idiom)

                    self.tgt_keywords[idiom] = self.preprocess_keywords(
                        marian_keywords.lower().split())
                    if not exclude_deepl:
                        self.tgt_keywords[idiom].update(
                            self.preprocess_keywords(
                                deepl_keywords.lower().split()))

                    self.src_keywords[idiom] = src_keywords.lower().split()

    def __call__(self, idiom: str, prd: str):
        """Return the classification of the translation for this idiom."""
        return self.classify(idiom, prd)

    def preprocess_keywords(self, keywords):
        unique_keywords = set()
        for keyword in keywords:
            unique_translations = set()
            for translation in keyword.split(";"):
                translation = translation.split('_')
                if len(translation) == 2:
                    translation1, translation2 = translation
                elif len(translation) == 3:
                    translation1, translation2, _ = translation
                else:
                    translation1 = translation[0]
                    translation2 = translation[0]
                translation1 = translation1.replace("&#39;", "'")
                translation2 = translation2.replace("&#39;", "'")
                if translation1.lower() and translation1.lower(): # self.nld_words[42:200]:
                    unique_translations.add(translation1.lower())
                if translation2.lower() and translation2.lower():
                    unique_translations.add(translation2.lower())
            unique_keywords.add(';'.join(list(unique_translations)))
        return unique_keywords

    def contains(self, idiom: str):
        """Check whether the classifier contains keywords for this idiom."""
        return idiom in self.tgt_keywords

    def classify(self, idiom: str, prd: str):
        """For a given idiom, predict the class of the predicted translation."""
        prd = prd.lower().strip()
        idiom = idiom.lower().strip()
        if idiom not in self.idioms:  # not self.tgt_keywords[idiom]:
            return "none"

        if self.use_spacy:
            lemmas = []
            for w in self.nlp_tgt(prd):
                lemmas.append(w.lemma_)
            prd = prd + " " + " ".join(lemmas)

        if idiom not in self.tgt_keywords:
            return "none"

        # Check if the Englis keyword is present
        keywords = self.src_keywords[idiom]
        keywords_present = []
        for keyword in keywords:
            keyword_present = []
            for w in keyword.split(';'):
                keyword_present.append(w in nltk.word_tokenize(prd))
            keywords_present.append(any(keyword_present))

        # If the keyword itself is present in the translation, return "word-by-word"
        if any(keywords_present):
            return "word-by-word"

        # Otherwise, check if the Dutch translation of the keyword is present
        keywords = self.tgt_keywords[idiom]
        keywords_present = []
        for keyword in keywords:
            keywords_present.append(
                any([w in prd for w in keyword.split(';')]))
        if any(keywords_present):
            return "word-by-word"

        # Otherwise, it is considered a paraphrase
        return "paraphrase"

    def colour_keyword(self, idiom: str, src: str):
        """Mark the idiom's keywords in the source sentence for the survey."""
        # First colour the words that are the exact keyword
        keywords = self.src_keywords[idiom]
        new_idiom = idiom
        for w in keywords:
            new_idiom = new_idiom.replace(
                w, f"<span style='color:red'><u>{w}</u></span>")

        # Then colour the words that are the lemmas of the keyword
        # inflected = []
        # for w in keywords:
        #     inflected.append(self.to_plural_singular(w)[1])
        # keywords = list(set(keywords + inflected))
        # print(keywords)
        keywords = [self.nlp_en(
            w)[0].lemma_ for w in keywords if w is not None]
        src_coloured = []
        for w in src.split():
            lemmas = [x.lemma_.lower() for x in self.nlp_en(w)]
            if any([w2 in l for l in lemmas for w2 in keywords]):
                w = f"<span style='color:red'><u>{w}</u></span>"
            src_coloured.append(w)
        return new_idiom, " ".join(src_coloured)

    def to_plural_singular(self, word):
        """Returns the plural form for singular inputs and vice versa."""
        tokens = self.nlp_en(word)
        if not inflect.singular_noun(tokens[0].text):
            return tokens[0]._.inflect('NNS')
        return tokens[0]._.inflect('NN')


if __name__ == "__main__":
    classifier = Classifier("idiom_keywords_translated_fr.tsv")
