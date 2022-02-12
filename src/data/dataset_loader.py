from torchtext.legacy.data import Example, Dataset

from data import ExampleUtil


class DatasetLoader:
    def __init__(self, source_field, target_field):
        self.fields = [
            ('source', source_field),
            ('target', target_field)
        ]

    def load(self, path):
        examples = [Example().fromlist(d, self.fields) for d in ExampleUtil.load(path)]
        return Dataset(examples, self.fields)
