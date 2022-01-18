import sys

sys.path.append('..')

from evaluation import *
import numpy as np
import pickle
import time

from tqdm.auto import tqdm


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
#
# class SkipGram:
#     def __init__(self, window_size, hidden_unit): #window_size: the number of input word, hidden_unit : dimension of vector
#         with open('./news/vocab.txt', 'rb') as f:
#             corpus = pickle.load(f)
#         self.word_to_id = corpus[0]
#         self.id_to_word = corpus[1]
#         self.vocab_size = len(self.word_to_id)
#         self.num_sentence = corpus[2]['<EOS>']
#         self.dimension = hidden_unit
#         self.lr = 0.025
#         self.window_size = np.random.randint(1, window_size)
#         self.cache = None
#
#         self.W_embedding = 0.01*np.random.randn(self.vocab_size, hidden_unit).astype('f')
#         self.W_embedding_b = np.zeros((self.vocab_size - 1, hidden_unit)).astype('f')
#         with open('./data/huffman.txt', 'rb') as hf:
#             (self.id_to_code, self.id_to_way) = pickle.load(hf)
#
#
    # def forward(self, contexts, center):
    #
    #     code = self.id_to_code[center]    #(V-1,)
    #     way = list(self.id_to_way[center])      #(V-1,)
    #     out = self.W_embedding[contexts]  #(window, D)
    #     score = code * np.dot(out, self.W_embedding_b[way].T)    #(window, V-1)
    #     self.cache = (contexts, out, score, code, way)
    #
    #     loss = np.sum(-np.log(sigmoid(score) + 1e-07))
    #     loss /= len(contexts)
    #
    #     return loss
#
#         out = self.W_embedding[center]
#         for context in contexts:
#             code = self.id_to_code[context]
#             way = list(self.id_to_way[context])
#             score = code * np.dot(self.W_embedding_b[way].T, out)
#
#
#     def backward(self, lr):
#         (contexts, out, score, code, way) = self.cache
#         dout = sigmoid(score)           #(window, V-1)
#         dout = code * (dout - 1)        #(window, V-1)
#         dx = np.dot(dout, self.W_embedding_b[way])     #(window, D)
#         dW_out = np.dot(dout.T, out)    #(V-1, D)
#
#         self.W_embedding_b[way] -= dW_out * lr
#         self.W_embedding[contexts] -= dx * lr
#
#         return None
#
#
#     def train(self, epoch):
#         start = time.time()
#         lr = self.lr
#         n = 0
#
#         for j in tqdm(range(epoch), desc='Epoch'):
#             loss = 0
#             count = 0
#
#             for i in tqdm(range(100), desc='Iteration'):
#                 if i < 10:
#                     num = '0' + str(i)
#                 else:
#                     num = str(i)
#                 data = './news/en-000' + num + '-of-00100.txt'
#                 with open(data, 'rb') as f:
#                     text = pickle.load(f)
#
#                 for sentence in text:
#                     n += 1
#                     #subsampling
#                     # new_sentence = self.subsampling.delete_vocab(id_to_word=self.id_to_word, sentence=sentence)
#                     contexts, target = create_contexts_target(sentence, window_size=self.window_size + 1)
#                     # contexts, target = create_contexts_target(new_sentence, window_size=self.window_size + 1)
#
#
#                     for i in range(len(contexts)):
#                         count += 1
#                         loss += self.forward(contexts[i], target[i])
#                         self.backward(lr)
#
#                     #lr decay
#                     alpha = 1 - n/(self.num_sentence * epoch)
#                     if alpha <= 0.0001:
#                         alpha = 0.0001
#
#                     lr = self.lr * alpha
#
#
#                     if count % 10000 == 1:
#                         train_time = (time.time() - start) / 3600
#                         avg_loss = loss/count
#                         print('time: {}, loss : {}'.format(train_time, avg_loss))
#                         print('{} sentence trained!'.format(n))
#                         print('learning rate : {}'.format(lr))
#
#                         count = 0
#                         loss = 0
#
#
#         print('Train Finished!')
#         print('Train time : {}'.format(time.time()-start))
#
#         with open('data/embedding_hf.pkl', 'wb') as f:
#             pickle.dump(self.W_embedding, f)
#         return None
#
#
# # train
with open('./news/vocab.txt', 'rb') as v:
    (word_to_id, id_to_word, count) = pickle.load(v)
with open('./data/huffman.txt', 'rb') as hf:
    (id_to_code, id_to_way) = pickle.load(hf)
with open('./news/en-00000-of-00100.txt', 'rb') as data:
    text = pickle.load(data)
    lr = 0.025
    W_embedding = np.random.uniform(low=-0.5, high=0.5, size=(len(id_to_code), 300))
    W_embedding_b = np.random.uniform(low=-0.5, high=0.5, size=(len(id_to_code) , 300))

    n=0
    total_loss = 0
    for sentence in tqdm(text):
        n += 1
        contexts, center = create_contexts_target(sentence, window_size=3)

        for i in range(len(center)):
            # hierarchical softmax
            loss = 0
            cache = []
            out = W_embedding[center[i]]      #(D)
            for context in contexts[i]:
                code = np.array(id_to_code[context])    #(V-1,)
                way = list(id_to_way[context])      #(V-1,)
                score = np.dot(code, np.dot(out, W_embedding_b[way].T))    #(window, V-1)
                loss += np.sum(-np.log(sigmoid(score) + 1e-07))
                cache.append((context, out, code, way, score))
            loss /= len(contexts[i])
            total_loss += loss

            #backward
            for (context, out, code, way, score) in cache:
                out = out.reshape(1, -1)                  #(1, D)
                dout = sigmoid(score) - 1
                dout = (dout * code).reshape(-1, 1)
                dx = np.dot(dout.T, W_embedding_b[way]).squeeze()     #(D)
                dW_out = np.dot(dout, out)    #(V-1, D)

                W_embedding_b[way] -= dW_out * lr
                W_embedding[context] -= dx * lr

        # lr decay
        if n % 10000 == 0:
            avg_loss = total_loss / n
            print('loss : {}'.format(avg_loss))

            n = 0
            total_loss = 0

#
#
# # #test
# # # save_test(file='./data/test.txt', word_to_id=word_to_id)
# # model = SkipGram(window_size=5, hidden_unit=300)
# # model.train(epoch=1)
# #
accuracy = accuracy(file='./data/test_labeled.pkl', W=W_embedding)
print(accuracy[0], accuracy[1])
print(most_similar('king', W_embedding, word_to_id, id_to_word, top=5))
