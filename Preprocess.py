import sys
sys.path.append('..')
import collections
import zipfile
import numpy as np
import time

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
        # dictionary = ['<UNK>']
        # for i in range(len(orders)):
        #     if orders[i][1] < self.min_count:
        #         del orders
        #         break
        #     else:
        #         dictionary.append(orders[i][0])
        #
        # return dictionary
        # dictionary = [['<UNK>', 0]]
        # for i in range(len(orders)):
        #     if orders[i][1] < self.min_count:
        #         dictionary[0][1] = len(orders) - i
        #         del orders
        #         break
        #     else:
        #         dictionary.append(orders[i])
        # return dictionary
        vocab, counts = ['<UNK>'], [0]
        for i in range(len(orders)):
            if orders[i][1] < self.min_count:
                counts[0] = len(orders) - i
                del orders
                break
            else:
                vocab.append(orders[i][0])
                counts.append(orders[i][1])

        # <UNK> 포함 sorting
        # for i in range(len(vocab)):
        #     if counts[i] < counts[i + 1]:
        #         temp1 = counts[i]
        #         counts[i] = counts[i + 1]
        #         counts[i + 1] = temp1
        #         temp2 = vocab[i]
        #         vocab[i] = vocab[i + 1]
        #         vocab[i + 1] = temp2
        #
        #     else:
        #         break

        return vocab, counts




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

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, iter=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = iter
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):

                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
