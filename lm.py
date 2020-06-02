from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from os.path import exists
import pickle

class LanguageModel:
    sequence_length = None

    def tokenize_sentences(self, data):
        """Split input collection of text to a collection of sentence collections"""
        print("Splitting sentences")
        filename = "data-sentsplit.pkl"

        if exists(filename):
            print("Loading")
            with open(filename, "rb") as f:
                res = pickle.load(f)
            return res

        res = []
        for text in tqdm(data):
            sentences = sent_tokenize(text)
            res.append(sentences)
        with open(filename, "wb") as f:
            pickle.dump(res, f)
        return res

class Dataset:
    """Dataset class to construct the required dataloaders"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
            return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
