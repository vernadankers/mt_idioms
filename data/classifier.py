import spacy
import dutch_words
from translate_keywords import to_plural_singular


class Classifier():
    """Simple classifier to label translations according to whether they
    contain the translation of a keywor (or not)."""

    def __init__(self, idiom_file: str):
        self.idioms = []
        self.nl_keywords = dict()
        self.en_keywords = dict()

        self.nlp_en = spacy.load("en_core_web_sm")
        self.nld_words = dutch_words.get_ranked()

        # Open the idioms that contain nouns from file
        with open(idiom_file, encoding="utf-8") as f_annot:
            for line in f_annot:
                if len(line.strip().split("\t")) != 5:
                    continue
                idiom, inclusion, _, en_keywords, nl_keywords = \
                    line.strip().split("\t")
                self.idioms.append(idiom)
                if inclusion == "yes":
                    self.nl_keywords[idiom] = nl_keywords.split()
                    self.en_keywords[idiom] = en_keywords.split()

    def __call__(self, idiom: str, prd: str):
        """Return the classification of the translation for this idiom."""
        return self.classify(idiom, prd)

    def contains(self, idiom: str):
        """Check whether the classifier contains keywords for this idiom."""
        return idiom in self.nl_keywords

    def classify(self, idiom: str, prd: str):
        """For a given idiom, predict the class of the predicted translation."""
        prd = prd.lower().strip()
        idiom = idiom.lower().strip()
        if idiom not in self.nl_keywords or not prd or "Voorzitter" in prd:
            return "none"

        # Check if the Englis keyword is present
        keywords = self.en_keywords[idiom]
        keywords_present = []
        for keyword in keywords:
            keyword_present = []
            for w in keyword.split(';'):
                # If the keyword is a Dutch word, we can't tell what the
                # category is...
                if w in self.nld_words:
                    return "none"
                keyword_present.append(w in prd)
            keywords_present.append(any(keyword_present))

        # If the keyword itself is present in the translation, return "copied"
        if any(keywords_present):
            return "copied"

        # Otherwise, check if the Dutch translation of the keyword is present
        keywords = self.nl_keywords[idiom]
        keywords_present = []
        for keyword in keywords:
            keywords_present.append(any([w in prd for w in keyword.split(';')]))
        if any(keywords_present):
            return "word-by-word"

        # Otherwise, it is considered a paraphrase
        return "paraphrase"

    def colour_keyword(self, idiom: str, src: str):
        """Mark the idiom's keywords in the source sentence for the survey."""
        # First colour the words that are the exact keyword
        keywords = self.en_keywords[idiom]
        new_idiom = idiom
        for w in keywords:
            new_idiom = new_idiom.replace(w, f"<span style='color:red'><u>{w}</u></span>")

        # Then colour the words that are the lemmas of the keyword
        inflected = []
        for w in keywords:
            inflected.append(to_plural_singular(w)[1])
        keywords = list(set(keywords + inflected))
        keywords = [self.nlp_en(w)[0].lemma_ for w in keywords if w is not None]
        src_coloured = []
        for w in src.split():
            lemmas = [x.lemma_.lower() for x in self.nlp_en(w)]
            if any([w2 in l for l in lemmas for w2 in keywords]):
                w = f"<span style='color:red'><u>{w}</u></span>"
            src_coloured.append(w)
        return new_idiom, " ".join(src_coloured)
