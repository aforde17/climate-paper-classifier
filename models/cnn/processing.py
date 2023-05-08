import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer

path="../../data/preprocessed_data.csv"
df = pd.read_csv(path)

class Processing():
    def __init__(self):
        self.df = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.vocab = pd.DataFrame()
        self.seq_len = 0
        self.num_words = 0

    def load(self):
        from sklearn import preprocessing
        self.df = pd.read_csv(path)
        self.df = self.df[['CLASS', 'DESCRIPTION']]
        self.df = self.df.dropna()
        self.df['DESCRIPTION'] = self.df['DESCRIPTION'].str.replace(r'<[^<>]*>', '', regex=True) # drop HTML tags

        le = preprocessing.LabelEncoder()
        le.fit(self.df['CLASS'])
        self.df['LABEL'] = le.transform(self.df['CLASS'])

        self.df = self.df[['LABEL', 'DESCRIPTION']]

        self.df['LABEL'].value_counts(dropna=False)

        self.text = self.df["DESCRIPTION"].values
        self.target = self.df["LABEL"].values
    
    def tokenize_and_build_vocabulary(self):
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        tokenizer = get_tokenizer('basic_english')

        def yield_tokens(train_iter):
            for text in train_iter:
                yield tokenizer(text)
 
        
        # tokenize and build vocab over all words 
        train_iter = iter(self.text.tolist())
        self.tokenized = list(map(lambda text : tokenizer(text), train_iter))
        # set length of longest sentence
        self.seq_len = max([len(x) for x in self.tokenized])
        # re-initialize train iter
        train_iter = iter(self.text.tolist())
        # build vocab
        self.vocab = build_vocab_from_iterator(
            yield_tokens(train_iter), specials=["<unk>"], max_tokens = self.seq_len)
        self.vocab.set_default_index(self.vocab['<unk>'])
        # set num words in vocab
        self.num_words = len(self.vocab)

    def word_to_idx(self):
        # Index representation	
        self.index_representation = list()

        for sentence in self.tokenized:
            temp_sentence = list()
            for word in sentence:
                idx = self.vocab.lookup_indices([word])
                temp_sentence.extend(self.vocab.lookup_indices([word]))
            self.index_representation.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        self.padded = list()
        print(len(self.index_representation))
        for sentence in self.index_representation:
            while len(sentence) < self.seq_len:
                sentence.append(pad_idx)
            self.padded.append(sentence)
        self.padded = np.array(self.padded)
    
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.padded, 
            self.target, 
            test_size=0.25, 
            random_state=30255
        )

    def process_all(self):
        self.load()
        self.tokenize_and_build_vocabulary()
        self.word_to_idx()
        self.padding_sentences()
        self.split_data()

    def prints(self):
        # Task 1
        print(f"Task 1: The number of words in the Vocab object is {len(self.vocab)}.")

        # # Task 2
        stoi_dict = self.vocab.get_stoi()
        word = "energy"
        print(f"Task 2: The index of the word '{word}' is {stoi_dict[word]}.")

        # # Task 3
        itos_dict = self.vocab.get_itos()
        idx = 500
        print(f"Task 3: The word at index 500 is '{itos_dict[idx]}'.")

        # # Task 4:
        word = "<unk>"
        print(f"Task 4: The index of the word '{word}' is {stoi_dict[word]}. Resetting default index to this value.")
