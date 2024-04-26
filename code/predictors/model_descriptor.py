from text_utils import transform_to_matrix
from pickle import dump, load

class ModelDescriptor:

    def __init__(self, model, features, ngram_type, ngrams_max_size, train_time):
        self.model = model
        self.features = features
        self.ngram_type = ngram_type
        self.ngrams_max_size = ngrams_max_size
        self.train_time = train_time

    def predict(self, texts, lemma_path):
        X = transform_to_matrix(texts, self.features, lemma_path, self.ngram_type, self.ngrams_max_size)
        yp = self.model.predict(X) 
        return yp

    def save_to_file(self, file_path):
        # Write the serialized object to a file
        with open(file_path, 'wb') as f:
            dump(self, f)

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, 'rb') as f:
            md = load(f)
        return md