from preprocessing import Preprocessor, Tokenization


def main():
    corpus = ["i'll be back", "hello my little friend"]
    tokenized_corpus = Tokenization().fit_transform(corpus)
    print(tokenized_corpus)    

if __name__=="__main__":
    main()