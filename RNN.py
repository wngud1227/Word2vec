import numpy as np
import ptb
from tqdm.auto import tqdm

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

class RNN:
    def __init__(self, W_x, W_h, b):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_b):
        W_x, W_h, b = self.params
        h = np.dot(x, W_x) + np.dot(h_b, W_h) + b
        h = np.tanh(h)
        self.cache = (x, h_b, h)
        return h

    def backward(self, dout):
        W_x, W_h, b = self.params
        (x, h_b, h) = self.cache
        dh = dout * (1 - h ** 2)
        db = np.sum(dh, axis=0)
        dW_x = np.dot(x.T, dh)
        dW_h = np.dot(h_b.T, dh)
        dh_b = np.dot(dh, W_h.T)

        self.grads[0] = dW_x
        self.grads[1] = dW_h
        self.grads[2] = db

        return dh_b

class TimeRNN:
    def __init__(self, W_x, W_h, b):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.cache = None
        self.layers = []
        self.h = []

    def forward(self, xs):
        W_x, W_h, b = self.params
        N, T, D = xs.shape
        _, H = W_h.shape
        h = np.zeros((N, H), dtype='f')
        for t in range(T):
            layer = RNN(W_x, W_h, b)
            h = layer.forward(xs[:,t,:], h)
            self.layers.append(layer)
        self.layers = self.layers[::-1]
        return h

    def backward(self, dhs):
        N, T, H = dhs.shape
        dh = 0
        for t in range(T):
            layer = self.layers[t]
            dh = layer.backward(dh + dhs[:,t,:])
            self.grads += layer.grads
        return dh

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.xs = None
        self.grads = [np.zeros_like(W)]

    def forward(self, xs):
        W, = self.params
        N, T = xs.shape
        V, D = W.shape
        self.xs = xs
        x_emb = np.zeros((N, T, D))
        for t in range(T):
            x_emb[:, t, :] = W[xs[:,t]]
        return x_emb

    def backward(self, dout):
        xs = self.xs
        grad = self.grads
        N, T = xs.shape
        for t in range(T):
            grad[xs[:, t]] += dout[:, t, :]
        self.grads = grad
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, xs):
        self.x = xs
        N, T = xs.shape
        W, b = self.params
        D, H = W.shape
        out = np.zeros((N, T, H))
        for t in range(T):
            out[:, t, :]= np.dot(xs[:, t, :], W) + b
        return out

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params
        db = np.zeros_like(b)
        dW = np.zeros_like(W)
        dx = np.zeros_like(x)
        for t in range(T):
            db += np.sum(dout[:, t, :], axis=0)
            dW += np.dot(x[:, t, :].T, dout[:, t, :])
            dx[:, t, :] = np.dot(dout[:, t, :], W.T)

        self.grads[0], self.grads[1] = dW, db
        return dx

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        if ts.dim == 3:
            ts = ts.argmax(axis=2)

        mask = (ts != -1)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask
        loss = -np.sum(ls) / mask.sum()
        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys
        dx[np.arange(N*T), ts] -= 1
        dx *= dout / mask.sum()
        dx *= mask[:, np.newaxis]
        dx = dx.reshape((N, T, V))
        return dx



class RNNLM:
    def __init__(self, V, D, H):
        # V : vocab size, D : dimension, H : hidden units
        W_em = np.random.rand(V, D)
        W_x = np.random.rand(D, H)
        W_h = np.random.rand(H, H)
        b = np.zeros(H)
        W_a = np.random.rand(H, V)
        b_a = np.zeros(V)

        self.layers = [
            TimeEmbedding(W_em),
            TimeRNN(W_x, W_h, b),
            TimeAffine(W_a, b_a)
        ]
        self.loss_layer = TimeSoftmaxWithLoss

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in self.layers:
            dout = layer.backward(dout)

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus = corpus[:1000]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]
ts = corpus[1:]

data_size = len(xs)
batch_size = 10
time_size = 5
max_iters = data_size // time_size
time_idx = 0

model = RNNLM(vocab_size, 100, 100)
jump = 999 // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in tqdm(range(100)):
    for iter in tqdm(range(max_iters)):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        loss = model.forward(batch_x, batch_t)
        model.backward()
        for i in range(len(model.params)):
            model.params[i] -= 0.1 * model.grads[i]
