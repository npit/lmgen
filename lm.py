from nltk.tokenize import sent_tokenize

class LanguageModel:
    sequence_length = None

    def tokenize_sentences(data):
        """Split input collection of text to a collection of sentence collections"""
        res = []
        for text in data:
            sentences = sent_tokenize(text)
            res.append(sentences)
        return text

class Dataset:
    """Dataset class to construct the required dataloaders"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
            return len(self.data)                                                                                                                                                        

    def __getitem__(self, index):
        return self.data[index]