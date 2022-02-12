import json


class ExampleUtil:
    @staticmethod
    def save(examples, path):
        with open(path, 'w') as f:
            f.write(json.dumps(len(examples)))
            f.write("\n")

            # Save examples
            for pair in examples:
                data = [pair.src, pair.trg]
                f.write(json.dumps(data))  # Write samples
                f.write("\n")

    @staticmethod
    def load(filename):
        examples = []
        with open(filename, 'r') as f:
            # Read num. elements (not really need it)
            total = json.loads(f.readline())

            # Save elements
            for i in range(total):
                line = f.readline()
                example = json.loads(line)
                examples.append(example)
        return examples
