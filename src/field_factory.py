from torchtext.legacy.data import Field

from spacy_tokenizer import SpacyTokenizer


class FieldFactory:

    @staticmethod
    def create(
            spacy_model,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True
    ):
        tokenizer = SpacyTokenizer(spacy_model)
        return Field(
            tokenize=tokenizer.tokenize,
            init_token=init_token,
            eos_token=eos_token,
            lower=lower
        )

    @classmethod
    def create_from_news_model(
            cls,
            language,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True
    ):
        return cls.create(f'{language}_core_news_sm')

    @classmethod
    def create_from_web_model(
            cls,
            language,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True
    ):
        return cls.create(f'{language}_core_web_sm')