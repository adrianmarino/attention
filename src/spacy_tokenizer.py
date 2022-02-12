import spacy


class SpacyTokenizer:
    def __init__(self, model_name):
        self.__model = spacy.load(model_name)

    def tokenize(self, text):
        return [token.text for token in self.__model.tokenizer(text)][::-1]
