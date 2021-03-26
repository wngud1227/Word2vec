import sys
sys.path.append('..')
import collections
import zipfile
import numpy as np

class vocab_word:
    def __init__(self, word):
        self.count = 0
        self.word = word
        self.code = 0
        self.codelen = 0

class preprocess:
    def __init__(self, file):
        self.file = file
        self.dictionary_size = 1000
        self.min_count = 5
        self.vocab_size = 0
        self.vocab_max_size = 1000
        self.Max_string = 50
        self.vocab = list(-1 for i in range(self.vocab_max_size))

    def readzip(self):
        with zipfile.ZipFile(self.file) as f:
            names = f.namelist()
            contents = f.read(names[0])
            text = contents.split()
            return text


    def readtext(self):
        newfile = []
        with open(self.file, 'r', encoding='UTF8') as f:
            for text in f.readlines():
                text = text.replace(' .', ' <EOS>')
                text = text.lower().split()
                for word in text:
                    if len(word) > self.Max_string:
                        text.remove(word)
                newfile.extend(text)
            return newfile

    def wordset(self):
        text = self.readtext()
        unique = collections.Counter(text)
        del text
        orders = unique.most_common()
        del unique
        dictionary = ['<UNK>']
        for i in range(len(orders)):
            if orders[i][1] < self.min_count:
                del orders
                break
            else:
                dictionary.append(orders[i][0])

        return dictionary

    # def Huffman_coding(self):
    #     vocab_size = len(self.wordset())
    #     pos1 =
    #     for i in range(len(self.wordset()) - 1):
    #         if

class UnigramSampler:
    def __init__(self, counts, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = len(counts)
        self.word_p = np.zeros(self.vocab_size)
        self.word_p = np.power(counts, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx
