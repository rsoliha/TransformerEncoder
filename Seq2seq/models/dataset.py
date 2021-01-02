from torchtext.data import Field, RawField
from torchtext.data import TabularDataset
from torchtext import data
import torch
import numpy as np
import random
from Seq2seq.models.uiconfig import *
def preprocessor(text):
    text = text.split()


def postprocessor(batch):
    b = torch.LongTensor(batch)
    return b.transpose(0,1)

def to_int(text):
    return int(text)


class UIdataset():
    def __init__(self):
        #Field to set properties of columns in dataset
        self.SEQ_X = Field(sequential= True,use_vocab=True, fix_length = UIconfig.seq_len)
        self.SEQ_Y = Field(sequential= True,use_vocab=True, fix_length = UIconfig.seq_len)


        self.datafields = [(None, None),(None, None),("seq_x", self.SEQ_X),("seq_y", self.SEQ_Y)]
        self.data_set = TabularDataset(
            path="../data/processed/Sequence_data_with_prefix.csv",
            format='csv',
            skip_header=True,
            fields=self.datafields
        )
        #make dictionary of unique words and map to integers
        self.SEQ_X.build_vocab(self.data_set.seq_x, self.data_set.seq_y)
        self.SEQ_Y.vocab = self.SEQ_X.vocab
        self.train_data, self.valid_data, self.test_data = self.data_set.split(split_ratio=[0.8,0.1,0.1], random_state=random.getstate())

    def get_vocab(self):
        return self.SEQ_X.vocab

    def get_vocab_item(self, index):
        return self.SEQ_X.vocab.itos[index]

    def get_train_data_ith(self,i):
        return vars(self.train_data[i])

    def get_vocab_size(self):
        return len(self.SEQ_X.vocab)

    #Get data in batches
    def get_loaders(self):
        train_iter, valid_iter = data.BucketIterator.splits(
            (self.train_data, self.valid_data),
            batch_size=UIconfig.batch_size,
            sort_within_batch=False,
            sort = False,
            shuffle = True,
            repeat = False )
        test_iter = data.Iterator(self.test_data, batch_size = UIconfig.batch_size,
                                  sort = False, sort_within_batch=False, repeat=False, shuffle=False)
        return train_iter, valid_iter, test_iter



def main():
    np.random.seed(UIconfig.seed)
    torch.manual_seed(UIconfig.seed)
    random.seed(UIconfig.seed)

    uidataset = UIdataset()
    print(uidataset.get_vocab_size())
    train_iter, valid_iter, test_iter = uidataset.get_loaders()

    for batch in train_iter:
        print(batch.seq_x)       # (seq_len, batch_size)
        print(batch.seq_y)
        break

if __name__ == "__main__":
    main()


