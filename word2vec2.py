import numpy as np
import pickle

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

    return np.array(contexts), np.array(target)

#Embedding layer
class Embedding:
    def __init__(self, W):
        self.params = W
        self.grads = np.zeros_like(W)
        self.idx = None

    def forward(self, idx):
        W = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW = self.grads
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
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

#optimizer
class SGD:
    def __init__(self, lr):
        self.lr == lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

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
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

class CBOW:
    def __init__(self, window_size, hidden_unit): #window_size: the number of input word, hidden_unit : dimension of vector
        with open('data/dataset/news/vocab.txt', 'r') as f:
            corpus = pickle.load(f)
        self.word_to_id = corpus[0]
        self.id_to_word = corpus[1]
        self.vocab = corpus[2]
        self.vocab_size == len(self.word_to_id)
        self.W_embedding = 0.01 * np.random.randn(self.vocab_size, hidden_unit).astype('f')
        self.W_embedding_b = 0.01 * np.random.randn(self.vocab_size, hidden_unit).astype('f')

        self.optimizer = SGD

        self.in_layers = [] #2 x window size 만큼의 embedding 저장
        for i in range(2 * window_size):
            layer = Embedding(self.W_embedding)
            self.in_layers.append(layer)

        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads



        # else:
        #     W_embedding = 0.01 * np.random.randn(V, H).astype('f')
        #     W_embedding_b = 0.01 * np.random.randn(H, V).astype('f')

    def forward(self, contexts, target): #input : (batch_size, hidden_unit)
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers) # h : batch size
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

