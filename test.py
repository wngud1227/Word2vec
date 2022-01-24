import sys

sys.path.append('..')

from evaluation import *
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

def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return contexts, target


class SkipGram:
    def __init__(self, window_size, hidden_unit): #window_size: the number of input word, hidden_unit : dimension of vector
        with open('news/vocab.txt', 'rb') as f:
            corpus = pickle.load(f)
        self.word_to_id = corpus[0]
        self.id_to_word = corpus[1]
        self.vocab_size = len(self.word_to_id)
        self.num_sentence = corpus[2]['<EOS>']
        self.dimension = hidden_unit
        self.lr = 0.025
        self.window_size = np.random.randint(1, window_size)
        self.cache = None
        self.x = None

        self.W_embedding = 0.01 * np.random.randn(self.vocab_size, self.dimension).astype('f')
        self.W_embedding_b = np.zeros((self.vocab_size - 1, self.dimension)).astype('f')


    def forward(self, contexts, center, codes, ways):
        loss = 0
        cache = []
        self.x = self.W_embedding[center].reshape(1, -1)        #(1, D)

        for context in contexts:
            code = np.array(codes[context]).reshape(-1, 1)     #(V-1, 1)
            way = ways[context]                 #(V-1)
            score = sigmoid(code * np.dot(self.W_embedding_b[way], self.x.T))             #(V-1, 1)
            loss -= np.sum(np.log(score + 1e-07))
            cache.append((context, code, way, score))
        self.cache = cache.copy()
        return loss/len(contexts)

    def backward(self, lr):
        l = len(self.cache)
        for (context, code, way, score) in self.cache:
            dout = code * (score - 1) / l                  #(V-1, 1)
            dx = np.dot(dout.T, self.W_embedding_b[way])                          #(1, D)
            dW_out = np.dot(dout, self.x)                          #(v-1, D)

            self.W_embedding_b[way] -= dW_out * lr
            self.W_embedding[context] -= dx.squeeze() * lr

        return None

    def train(self, epoch):
        start = time.time()
        lr = self.lr
        n = 0
        with open('./data/huffman_new.txt', 'rb') as hf:
            (id_to_code, id_to_way) = pickle.load(hf)

        for j in tqdm(range(epoch), desc='Epoch'):
            loss = 0
            count = 0

            # for i in tqdm(range(100), desc='Iteration'):
            #     if i < 10:
            #         num = '0' + str(i)
            #     else:
            #         num = str(i)
            data = './news/en-00000-of-00100.txt'
            with open(data, 'rb') as f:
                text = pickle.load(f)

            for sentence in tqdm(text):
                n += 1
                contexts, target = create_contexts_target(sentence, window_size=self.window_size + 1)


                for i in range(len(contexts)):
                    count += 1
                    loss += self.forward(contexts[i], target[i], id_to_code, id_to_way)
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

        with open('data/embedding_sg_hf.pkl', 'wb') as f:
            pickle.dump(self.W_embedding, f)
        return None

#
#
# # #test
# save_test(file='./data/test.txt', word_to_id=word_to_id)
# model = SkipGram(window_size=5, hidden_unit=300)
# model.train(epoch=1)

with open('./data/embedding_sg_hf.pkl', 'rb') as f:
    W_embedding = pickle.load(f)
accuracy = accuracy(file='./news/test_labeled.pkl', W=W_embedding)
print(accuracy[0], accuracy[1])
#
# print(most_similar('king', W_embedding, word_to_id, id_to_word, top=5))
