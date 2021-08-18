import numpy as np
import pickle
import time
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

#subsampling
class subsampling:
    def __init__(self, counts, t):
        value = list(counts.values())
        self.whole_vocab = np.sum(value)
        self.del_p = {}
        for (word, f) in counts.items():
            temp = f / self.whole_vocab
            self.del_p[word] = 1 - np.sqrt(t / temp)

    def delete_vocab(self, id_to_word, sentence):
        new_sentence = []
        for word in sentence:
            if self.del_p[id_to_word[word]] > np.random.rand():
                continue
            else:
                new_sentence.append(word)
        return new_sentence

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


#Negative Sampling
class UnigramSampler:
    def __init__(self, word_to_id, counts, power, sample_size): #counts = corpus[2](=self.vocab)
        self.sample_size = sample_size
        self.vocab_size = len(counts)

        p = []
        for word in counts.keys():
            freq = int(pow(counts[word], power))
            p.extend([word_to_id[word]] * freq)
        self.word_p = np.array(p)

    def get_negative_sample(self, target):
        negative_sample = []
        while(1):
            sample = np.random.randint(low=0, high=self.word_p.shape[0])
            if target == self.word_p[sample]:
                continue
            else:
                negative_sample.append(self.word_p[sample])

            if len(negative_sample) >= self.sample_size:
                break
        return negative_sample

#activation
def sigmoid(x):
    out = 1./(1.+np.exp(-x))
    return out

#CBOW
class CBOW:
    def __init__(self, window_size, hidden_unit, hs=0): #window_size: the number of input word, hidden_unit : dimension of vector
        with open('data/dataset/news/vocab.txt', 'rb') as f:
            corpus = pickle.load(f)
        self.word_to_id = corpus[0]
        self.id_to_word = corpus[1]
        self.vocab_size = len(self.word_to_id)
        self.num_sentence = corpus[2]['<EOS>']
        self.dimension = hidden_unit
        self.lr = 0.025
        # self.window_size = window_size
        self.window_size = np.random.randint(1, window_size)
        self.cache = None
        self.subsampling = subsampling(counts=corpus[2], t=0.00001)

        if hs == 0: #negative sampling
            self.Unigram = UnigramSampler(self.word_to_id, corpus[2], power=0.75, sample_size=5)
            self.W_embedding = 0.01 * np.random.randn(self.vocab_size, self.dimension).astype('f')
            self.W_embedding_b = np.zeros((self.vocab_size, self.dimension)).astype('f')

        # else: #hierarchical softmax
        #     W_embedding = 0.01 * np.random.randn(V, H).astype('f')
        #     W_embedding_b = 0.01 * np.random.randn(H, V).astype('f')

    def forward(self, contexts, target):

        x = self.W_embedding[contexts]
        x = np.sum(x, axis=0)
        x /= len(contexts)
        x = x.reshape(1, self.dimension)

        negative_sample = self.Unigram.get_negative_sample(target)

        label = np.append([target], negative_sample)
        out = sigmoid(np.dot(x, self.W_embedding_b[label].T))

        p_loss = -np.log(out[:, :1] + 1e-07)
        n_loss = -np.sum(np.log(1 - out[:, 1:] + 1e-07))
        self.cache = (x, out, label)

        return p_loss + n_loss



    def backward(self):
        (x, out, label) = self.cache
        dout = out.copy()
        dout[:, :1] -= 1
        dW_out = np.dot(x.T, dout).T
        dx = np.dot(dout, self.W_embedding_b[label])

        return dx, dW_out

    def train(self, epoch):
        start = time.time()
        lr = self.lr
        n = 0

        for j in tqdm(range(epoch), desc='Epoch'):
            loss = 0
            count = 0

            for i in tqdm(range(100), desc='Iteration'):
                if i < 10:
                    num = '0' + str(i)
                else:
                    num = str(i)
                data = 'data/dataset/news/en-000' + num + '-of-00100.txt'
                with open(data, 'rb') as f:
                    text = pickle.load(f)

                for sentence in text:
                    n += 1
                    #subsampling
                    new_sentence = self.subsampling.delete_vocab(id_to_word=self.id_to_word, sentence=sentence)
                    contexts, target = create_contexts_target(new_sentence, window_size=self.window_size + 1)

                    for i in range(len(contexts)):
                        count += 1
                        loss += self.forward(contexts[i], target[i])
                        dx, dW_out = self.backward()

                        self.W_embedding_b[self.cache[2]] -= dW_out * lr
                        self.W_embedding[contexts[i]] -= dx.squeeze() / len(contexts[i]) * lr

                    #lr decay
                    alpha = 1 - n/(self.num_sentence * epoch)
                    if alpha <= 0.0001:
                        alpha = 0.0001

                    lr = self.lr * alpha


                    if count % 10000 == 1:
                        train_time = (time.time() - start) / 3600
                        avg_loss = loss/count
                        print('time: {}, loss : {}'.format(train_time, avg_loss))
                        print('{} sentence trained!'.format(n))

                        count = 0
                        loss = 0


        print('Train Finished!')
        print('Train time : {}'.format(time.time()-start))

        with open('data/dataset/embedding_sub.pkl', 'wb') as f:
            pickle.dump(self.W_embedding, f)
        return None

#Skip-Gram
class SkipGram:
    def __init__(self, window_size, hidden_unit, hs=0): #window_size: the number of input word, hidden_unit : dimension of vector
        with open('data/dataset/news/vocab.txt', 'rb') as f:
            corpus = pickle.load(f)
        self.word_to_id = corpus[0]
        self.id_to_word = corpus[1]
        self.vocab_size = len(self.word_to_id)
        self.num_sentence = corpus[2]['<EOS>']
        self.dimension = hidden_unit
        self.lr = 0.025
        self.window_size = np.random.randint(1, window_size)
        self.cache = None
        self.subsampling = subsampling(counts=corpus[2], t=0.00001)

        if hs == 0: #negative sampling
            self.NEG = True
            self.Unigram = UnigramSampler(self.word_to_id, corpus[2], power=0.75, sample_size=5)
            self.W_embedding = np.random.uniform(low=-0.5, high=0.5, size=(self.vocab_size, self.dimension))
            self.W_embedding_b = np.zeros((self.vocab_size, self.dimension)).astype('f')

        # else: #hierarchical softmax
        #     W_embedding = 0.01 * np.random.randn(V, H).astype('f')
        #     W_embedding_b = 0.01 * np.random.randn(H, V).astype('f')

    def forward(self, contexts, target):
        if self.NEG:
            loss = 0
            cache = []

            for context in contexts:
                x = self.W_embedding[context]
                x = x.reshape(1, self.dimension)

                negative_sample = self.Unigram.get_negative_sample(target)
                label = np.append([target], negative_sample)
                out = sigmoid(np.dot(x, self.W_embedding_b[label].T))

                p_loss = -np.log(out[:, :1] + 1e-07)
                n_loss = -np.sum(np.log(1 - out[:, 1:] + 1e-07))
                loss += (p_loss + n_loss)
                cache.append((x, context, label, out))

            self.cache = cache
            return float(loss)/len(contexts)

    def backward(self, lr):
        if self.NEG:
            for x, context, label, out in self.cache:
                dout = out.copy()
                dout[:, :1] -= 1
                dW_out = np.dot(x.T, dout).T
                dx = np.dot(dout, self.W_embedding_b[label])

                self.W_embedding_b[label] -= dW_out * lr
                self.W_embedding[context] -= dx.squeeze() * lr

            return None

    def train(self, epoch):
        start = time.time()
        lr = self.lr
        n = 0

        for j in tqdm(range(epoch), desc='Epoch'):
            loss = 0
            count = 0

            for i in tqdm(range(100), desc='Iteration'):
                if i < 10:
                    num = '0' + str(i)
                else:
                    num = str(i)
                data = 'data/dataset/news/en-000' + num + '-of-00100.txt'
                with open(data, 'rb') as f:
                    text = pickle.load(f)

                for sentence in text:
                    n += 1
                    #subsampling
                    new_sentence = self.subsampling.delete_vocab(id_to_word=self.id_to_word, sentence=sentence)
                    # contexts, target = create_contexts_target(sentence, window_size=self.window_size + 1)
                    contexts, target = create_contexts_target(new_sentence, window_size=self.window_size + 1)


                    for i in range(len(contexts)):
                        count += 1
                        loss += self.forward(contexts[i], target[i])
                        self.backward(lr)

                    #lr decay
                    alpha = 1 - n/(self.num_sentence * epoch)
                    if alpha <= 0.0001:
                        alpha = 0.0001

                    lr = self.lr * alpha


                    if count % 10000 == 1:
                        train_time = (time.time() - start) / 3600
                        avg_loss = loss/count
                        print('time: {}, loss : {}'.format(train_time, avg_loss))
                        print('{} sentence trained!'.format(n))
                        print('learning rate : {}'.format(lr))

                        count = 0
                        loss = 0


        print('Train Finished!')
        print('Train time : {}'.format(time.time()-start))

        with open('data/dataset/embedding_sg1.pkl', 'wb') as f:
            pickle.dump(self.W_embedding, f)
        return None
