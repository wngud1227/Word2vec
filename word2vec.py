import sys
sys.path.append('..')
import numpy as np
import time
from Preprocess import preprocess, UnigramSampler, SigmoidWithLoss


def create_contexts_target(corpus, window_size=5):
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

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        # out = []
        # for i in idx:
        #     out.append(W[i])
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

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size) #corpus = counts
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):

        batch_size = target.shape
        negative_sample = self.sampler.get_negative_sample(target) #(batch, sample size)

        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh

class CBOW:
    def __init__(self, window_size, negative_sample, corpus, W_in, W_out):
        if negative_sample:
            self.W_embedding = W_in
            self.W_embedding_b = W_out

            self.in_layers = [] #2 x window size 만큼의 embedding 저장
            for i in range(2 * window_size):
                layer = Embedding(W_embedding)
                self.in_layers.append(layer)
            self.ns_loss = NegativeSamplingLoss(W_embedding_b, corpus, power=0.75, sample_size=5)

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


file = 'data/news1.txt'

max_epoch = 6
max_window_size = 5
hidden_unit = 200
batch_size = 100


preprocess = preprocess(file)
vocab, counts = preprocess.wordset()
vocab_size = len(vocab)
huffman = []
start_time = time.time()

W_embedding = 0.01 * np.random.randn(vocab_size, hidden_unit).astype('f')
W_embedding_b = 0.01 * np.random.randn(vocab_size, hidden_unit).astype('f')

# W_embedding_b = 0.01 * np.random.randn(hidden_unit, 1).astype('f')
total_loss, loss_count = 0, 0

with open(file, 'r', encoding='UTF8') as f:
    for epoch in range(max_epoch):
        iter = 0
        window_size = max_window_size
        # window_size = np.random.randint(1, max_window_size)
        x, t = [], []
        for sentence in f.readlines():
            sentence = sentence.replace(' .', ' <EOS>')
            sentence = sentence.lower().split()
            data = []
            for word in sentence:
                if word in vocab:
                    data.append(vocab.index(word))
                else:
                    data.append(0)
            # data size : words in a sentence
            contexts, target = create_contexts_target(data, window_size)
            idx = np.random.permutation(np.arange(len(contexts)))
            x.extend(contexts[idx])
            t.extend(target[idx])
            
            while len(t) >= batch_size:
                model = CBOW(window_size, negative_sample=True, corpus=counts, W_in=W_embedding, W_out=W_embedding_b)
                batch_x = np.array(x[:batch_size]) #batch size x window size
                batch_t = np.array(t[:batch_size])

                x = x[batch_size:]
                t = t[batch_size:]

                loss = model.forward(batch_x, batch_t)
                model.backward()
                iter += 1

                elapsed_time = time.time() - start_time
                print('Epoch: {}, Iteration: {} , Time: {}, loss: {}'.format(epoch + 1, iter, elapsed_time, loss))

                W_embedding = model.W_embedding
                W_embedding_b = model.W_embedding_b



