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
        dx = np.dot(dh, W_x.T)

        self.grads[0][...] = dW_x
        self.grads[1][...] = dW_h
        self.grads[2][...] = db

        return dx, dh_b

class TimeRNN:
    def __init__(self, W_x, W_h, b):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.layers = None
        self.h = []

    def forward(self, xs):
        W_x, W_h, b = self.params
        N, T, D = xs.shape
        _, H = W_h.shape

        self.layers = []
        h = np.zeros((N, H), dtype='f')
        hs = np.zeros((N, T, H), dtype='f')
        for t in range(T):
            layer = RNN(W_x, W_h, b)
            h = layer.forward(xs[:,t,:], h)
            hs[:, t, :] = h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        W_x, W_h, b = self.params
        D, H = W_x.shape
        N, T, H = dhs.shape
        dxs = np.zeros((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dh + dhs[:,t,:])
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        return dxs

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.xs = None

    def forward(self, xs):
        self.xs = xs
        W, = self.params
        N, T = xs.shape
        V, D = W.shape
        x_emb = np.zeros((N, T, D), dtype='f')
        for t in range(T):
            x_emb[:, t, :] = W[xs[:,t]]
        return x_emb

    def backward(self, dout):
        W, = self.params
        N, T, D = dout.shape
        grad = np.zeros_like(W)
        for t in range(T):
            grad[self.xs[:, t]] += dout[:, t, :]
        self.grads[0][...] = grad
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, xs):
        self.x = xs
        N, T, H = xs.shape
        W, b = self.params

        rx = xs.reshape(N*T, -1)
        out= np.dot(rx, W) + b

        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = self.x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T).reshape(N, T, -1)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelㅇㅔ 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx



class RNNLM:
    def __init__(self, V, D, H):
        # V : vocab size, D : dimension, H : hidden units
        W_em = np.random.rand(V, D).astype('f') / 100
        W_x = np.random.rand(D, H).astype('f') / np.sqrt(D/2)
        W_h = np.random.rand(H, H).astype('f') / np.sqrt(H/2)
        b = np.zeros(H)
        W_a = np.random.rand(H, V).astype('f') / np.sqrt(H/2)
        b_a = np.zeros(V)

        self.layers = [
            TimeEmbedding(W_em),
            TimeRNN(W_x, W_h, b),
            TimeAffine(W_a, b_a)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

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
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

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
total_loss = 0
loss_count = 0
ppl_list = []

model = RNNLM(vocab_size, 100, 100)
jump = (len(corpus) - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in tqdm(range(100)):
    # for iter in tqdm(range(max_iters)):
    for iter in range(max_iters):
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
        total_loss += loss
        loss_count += 1

    ppl = np.exp(total_loss / loss_count)
    print(' | epoch %d | perplexity %.2f' % (epoch+1, ppl))
    total_loss, loss_count = 0, 0
    ppl_list.append(float(ppl))

import matplotlib.pyplot as plt

plt.plot(ppl_list)
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('He Result')
plt.savefig('He.png')
plt.show()
