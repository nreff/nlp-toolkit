from tqdm import tqdm

import spacy
import en_core_web_sm
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score

from preprocessing import Preprocessor, Tokenization
from vectorization import Vectorizer, DimReducer
from clustering import Clusterizer
from visualization import Visualizer

    


def newsgroups_20():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    tokenizer = Tokenization(disable=['ner'])
    newsgroups.data = [tokenizer.fit_transform(text) for text in tqdm(newsgroups.data)]

    vectorizer = Vectorizer('tfidf')
    vectors = vectorizer.fit_transform(newsgroups.data)

    n_dimensions = 300            
    dim_reducer =  DimReducer("SVD", n_dimensions)           
    vectors_reduced = dim_reducer.fit_transform(vectors)

    clusters = Clusterer('kmeans', n_clusters=20).fit_transform(vectors_reduced)  

    Visualizer('tsne').fit_transform(vectors_reduced, clusters)

    score = adjusted_rand_score(newsgroups.target, clusters)
    print("Adjusted Rand Score:", score)


def main():
    corpus = ["i'll be back george the bear", "hello my little friend changing changes changed"]
    tokenizer = Tokenization(disable=['ner', 'lemmatizer', 'attribute_ruler', 'parser', 'tagger'])
    print(tokenizer.disabled_pipeline_components)
    tokenized_corpus = tokenizer.fit_transform(corpus)
    print(f"Preprocessed corpus without lemmatization {tokenized_corpus}")  

    print('-'*10)
    tokenizer = Tokenization(disable=['ner'])
    print(tokenizer.disabled_pipeline_components)
    print('-'*10)

    tokenized_corpus_lemmatized = tokenizer.fit_transform(corpus)
    print(f"Preprocessed corpus with lemmatization {tokenized_corpus_lemmatized}")

if __name__=="__main__":
        
    #main()
    newsgroups_20()