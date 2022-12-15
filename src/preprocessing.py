from typing import List

import spacy



class Preprocessor():
    def __init__(self, doc: List[str]):
        self.is_fit = False
        
    def fit_transform():
        self.is_fit = True
        return self



class Tokenization():
    def __init__(self, disable=None):
        if disable is not None:
            self.disable = disable
        else:
            self.disable = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
        
    def fit_transform(self, corpus: List[str]):
        nlp = spacy.load("en_core_web_sm")
        return [[token.text for token in doc] for doc in nlp.pipe(corpus, disable=self.disable)]