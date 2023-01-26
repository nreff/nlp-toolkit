from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
# from gensim.models import Word2Vec, Doc2Vec
# from gensim.models.doc2vec import TaggedDocument

# import torch
# from transformers import BertModel, BertTokenizer

class Vectorizer:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertModel.from_pretrained('bert-base-uncased')

    def fit_transform(self, data):
        if self.method == 'count':
            vectorizer = CountVectorizer(**self.kwargs)
            return vectorizer.fit_transform(data)
        elif self.method == 'tfidf':
            vectorizer = TfidfVectorizer(**self.kwargs)
            return vectorizer.fit_transform(data)
        else: 
            raise ValueError("Invalid method. Choose 'count', 'tfidf'.")   
        # elif self.method == 'word2vec':
        #     model = Word2Vec(data, **self.kwargs)
        #     return model
        # elif self.method == 'doc2vec':
        #     tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        #     model = Doc2Vec(tagged_data, **self.kwargs)
        #     return model
        # elif self.method == 'bert':
        #     input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True) for text in data])
        #     with torch.no_grad():
        #         last_hidden_states = self.model(input_ids)[0]
        #     return last_hidden_states.numpy()
        # else:
        #     raise ValueError("Invalid method. Choose 'count', 'tfidf', 'word2vec', 'doc2vec' or 'bert'.")


class DimReducer:
    def __init__(self, method, n_components):
        self.method = method
        self.n_components = n_components

    def fit_transform(self, data):
        if self.method == 'LDA':
            lda = LatentDirichletAllocation(n_components=self.n_components)
            return lda.fit_transform(data)
        elif self.method == 'SVD':
            svd = TruncatedSVD(n_components=self.n_components)
            return svd.fit_transform(data)
        else:
            raise ValueError("Invalid method. Choose 'LDA' or 'SVD'.")