import numpy as np
import pickle
from tqdm.auto import tqdm

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list
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

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads

#context & target
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return contexts, target

#Embedding layer
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

#optimizer
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):

        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

#loss
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
        # batch_size = self.t.shape[0]
        #
        # dx = (self.y - self.t) * dout / batch_size
        dx = (self.y - self.t) * dout
        return dx

#Negative Sampling
class UnigramSampler:
    def __init__(self, counts, power, sample_size): #counts = corpus[2](=self.vocab)
        self.sample_size = sample_size
        self.vocab_size = len(counts)
        p = []
        for word in counts.keys():
            p.append(counts[word])
        self.word_p = np.power(p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        p = self.word_p.copy()
        p[target] = 0
        p /= p.sum()
        negative_sample = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        negative_sample = self.sampler.get_negative_sample(target)
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = 1
        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = 0
        for i in range(self.sample_size):
            negative_target = negative_sample[i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh

#CBOW
class CBOW:
    def __init__(self, window_size, hidden_unit, hs=0): #window_size: the number of input word, hidden_unit : dimension of vector
        with open('data/dataset/news/vocab.txt', 'rb') as f:
            corpus = pickle.load(f)
        self.word_to_id = corpus[0]

        self.id_to_word = corpus[1]
        self.vocab = corpus[2]
        self.vocab_size = len(self.word_to_id)
        self.W_embedding = 0.01 * np.random.randn(self.vocab_size, hidden_unit).astype('f')
        self.W_embedding_b = 0.01 * np.random.randn(self.vocab_size, hidden_unit).astype('f')

        self.optimizer = SGD(lr=0.025)
        self.window_size = window_size
        # self.window_size = np.random.randint(1, window_size)

        if hs == 0: #negative sampling
            self.Unigram = UnigramSampler(corpus[2], power=0.75, sample_size=5)
            self.in_layers = [] #2 x window size 만큼의 embedding 저장
            for i in range(2 * self.window_size):
                layer = Embedding(self.W_embedding)
                self.in_layers.append(layer)
            self.ns_loss = NegativeSamplingLoss(self.W_embedding_b, corpus[2], power=0.75, sample_size=5)
            layers = self.in_layers + [self.ns_loss]

            self.params, self.grads = [], []
            for layer in layers:
                self.params += layer.params
                self.grads += layer.grads

        # else: #hierarchical softmax
        #     W_embedding = 0.01 * np.random.randn(V, H).astype('f')
        #     W_embedding_b = 0.01 * np.random.randn(H, V).astype('f')

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss


    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

    def train(self, epoch):
        for j in tqdm(range(epoch), desc='Epoch'):
            iter = 0
            for i in tqdm(range(100), desc='Iteration'):
                if i < 10:
                    num = '0' + str(i)
                else:
                    num = str(i)
                data = 'data/dataset/news/en-000' + num + '-of-00100.txt'
                with open(data, 'rb') as f:
                    text = pickle.load(f)

                n=0
                for sentence in text:
                    n += 1
                    loss = 0
                    contexts, target = create_contexts_target(sentence, window_size=self.window_size + 1)

                    for i in range(len(contexts)):
                        loss += self.forward(contexts[i], target[i])
                        self.backward()
                        self.params, self.grads = remove_duplicate(self.params, self.grads)
                        self.optimizer.update(self.params, self.grads)

                    print('{} sentence'.format(n))
                    print(loss)
