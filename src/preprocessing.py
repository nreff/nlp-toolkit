from typing import List

import spacy



class Preprocessor():
    def __init__(self, doc: List[str]):
        self.is_fit = False
        
    def fit_transform():
        self.is_fit = True
        return self



class Tokenization():
    def __init__(self, disable: List[str]):
        self.nlp = spacy.load("en_core_web_sm")
        if disable:
            self.nlp.disable_pipes(disable)
        self._disabled_pipeline_components = disable
        
    def preprocess(self, text: str) -> str: 
        doc = self.nlp(text)
        if 'lemmatizer' in self.disabled_pipeline_components:
            return  " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        
    def fit_transform(self, corpus: List[str]) -> List[str]:
        return [self.preprocess(text) for text in corpus]

    @property
    def disabled_pipeline_components(self) -> List[str]:
        return self._disabled_pipeline_components
    
    @property
    def pipe_names(self) -> List[str]:
        return self.nlp.pipe_names
