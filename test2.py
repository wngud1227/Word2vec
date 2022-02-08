import pickle
import numpy as np
from evaluation import *
import time
from tqdm.auto import tqdm

def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out

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

class subsampling:
    def __init__(self, counts, id_to_word, t=1e-05):
        value = list(counts.values())
        self.whole_vocab = np.sum(value)
        self.id_to_word = id_to_word
        self.del_p = {}
        for (word, f) in counts.items():
            temp = f / self.whole_vocab
            self.del_p[word] = 1 - np.sqrt(t / temp)

    def delete_vocab(self, sentence):
        new_sentence = []
        for word in sentence:
            if self.del_p[self.id_to_word[word]] > np.random.rand():
                continue
            else:
                new_sentence.append(word)
        return new_sentence

class UnigramTable:
    def __init__(self, vocab, word_to_id, power=0.75):
        self.table = []
        for word in vocab.keys():
            id = word_to_id[word]
            count = int(vocab[word] ** power)
            self.table.extend([id] * count)
        self.p_sum = len(self.table)

    def negative_sampling(self, target, num_sample=5):
        neg = [target]
        while(1):
            neg_sample = self.table[np.random.randint(0, self.p_sum)]
            if neg_sample in neg:
                continue
            neg.append(neg_sample)
            if len(neg) > num_sample:
                break
        return neg

# class Skipgram:
#     def __init__(self, window_size=5, sample_size=5, dimension=300):
#         with open('news/vocab.txt', 'rb') as v:
#             (word_to_id, id_to_word, vocab) = pickle.load(v)
#         self.num_sentence = vocab['<EOS>']
#         self.W = 0.01 * np.random.randn(len(vocab), dimension).astype('f')
#         self.W_b = np.zeros((len(vocab), dimension)).astype('f')
#         self.unigram = UnigramTable(vocab, word_to_id, 0.75)
#         self.sample_size = sample_size
#         self.window_size = np.random.randint(1, window_size+1)
#         self.subsampling = subsampling(vocab, id_to_word)
#
#     def train(self, contexts, center, lr):
#         loss = 0
#
#         for context in contexts:
#             x = self.W[context].reshape(1, -1)
#             target = self.unigram.negative_sampling(center, self.sample_size)
#             score = sigmoid(np.dot(x, self.W_b[target].T))
#             p_loss = -np.log(score[:, :1] + 1e-07)
#             n_loss = -np.sum(np.log(1 - score[:, 1:] + 1e-07))
#             loss += (p_loss + n_loss)
#
#             score[:, :1] -= 1 #(1, S)
#             self.W[context] -= lr * np.dot(score, self.W_b[target]).squeeze()
#             self.W_b[target] -= lr * np.dot(score.T, x)
#         return loss / len(contexts)
#
# start = time.time()
# lr = 0.025
# n = 0
# epoch = 1
# model = Skipgram()
# for j in tqdm(range(epoch), desc='Epoch'):
#     count = 0
#
#     for i in tqdm(range(100), desc='Iteration'):
#         if i < 10:
#             num = '0' + str(i)
#         else:
#             num = str(i)
#         data = 'news/news.en-000' + num + '-of-00100'
#         with open(data, 'rb') as f:
#             text = pickle.load(f)
#
#         for sentence in tqdm(text):
#             n += 1
#             # contexts, target = create_contexts_target(sentence, window_size=model.window_size)
#             new_sentence = model.subsampling.delete_vocab(sentence)
#             contexts, target = create_contexts_target(new_sentence, window_size=model.window_size)
#
#             for i in range(len(contexts)):
#                 count += 1
#                 loss = model.train(contexts[i], target[i], lr)
#
#             #lr decay
#             alpha = 1 - n/(model.num_sentence * epoch)
#             if alpha <= 0.0001:
#                 alpha = 0.0001
#
#             lr = lr * alpha
#
#             if count % 1000 == 1:
#                 train_time = (time.time() - start) / 3600
#                 print('time: {}, loss : {}'.format(train_time, loss))
#                 print('{} sentence trained!'.format(n))
#                 print('learning rate : {}'.format(lr))
#
#                 count = 0
#
#
#
# print('Train Finished!')
# print('Train time : {}'.format(time.time()-start))
#
# with open('data/embedding_sg_neg5.pkl', 'wb') as f:
#     pickle.dump(model.W, f)

class SkipGram:
    def __init__(self, window_size=5, hidden_unit=300): #window_size: the number of input word, hidden_unit : dimension of vector
        with open('news/vocab.txt', 'rb') as f:
            corpus = pickle.load(f)
        self.word_to_id = corpus[0]
        self.id_to_word = corpus[1]
        self.vocab = corpus[2]
        self.vocab_size = len(self.word_to_id)
        self.num_sentence = self.vocab['<EOS>']
        self.dimension = hidden_unit
        self.lr = 0.025
        self.window_size = np.random.randint(1, window_size)
        self.cache = None
        self.subsampling = subsampling(self.vocab, self.id_to_word)
        self.Unigram = UnigramTable(self.vocab, self.word_to_id)
        self.W_embedding = np.random.uniform(low=-0.5, high=0.5, size=(self.vocab_size, self.dimension))
        self.W_embedding_b = np.zeros((self.vocab_size, self.dimension)).astype('f')

    def forward(self, contexts, target):
        loss = 0
        cache = []
        for context in contexts:
            x = self.W_embedding[context]
            x = x.reshape(1, self.dimension)

            negative_sample = self.Unigram.negative_sampling(target)
            label = np.append([target], negative_sample)
            out = sigmoid(np.dot(x, self.W_embedding_b[label].T))

            p_loss = -np.log(out[:, :1] + 1e-07)
            n_loss = -np.sum(np.log(1 - out[:, 1:] + 1e-07))
            loss += (p_loss + n_loss)
            cache.append((x, context, label, out))

        self.cache = cache
        return float(loss)/len(contexts)

    def backward(self, lr):
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
                data = './news/news.en-000' + num + '-of-00100'
                with open(data, 'rb') as f:
                    text = pickle.load(f)

                for sentence in tqdm(text):
                    n += 1
                    #subsampling
                    new_sentence = self.subsampling.delete_vocab(sentence=sentence)
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

        with open('data/embedding_sg_neg5.pkl', 'wb') as f:
            pickle.dump(self.W_embedding, f)
        return None

model = SkipGram()
model.train(1)
# file = 'data/test.txt'
# with open('news/vocab.txt', 'rb') as v:
#     (word_to_id, id_to_word, vocab) = pickle.load(v)
#     print(len(vocab))
#     test = make_test(file)
#     print(len(test[1]))
#     save_test(file, word_to_id)
#
# with open('data/embedding_sg_neg5.pkl', 'rb') as f:
#     W_embedding = pickle.load(f)
# accuracy = accuracy(file='./news/test_labeled.pkl', W=W_embedding)
# print(accuracy[0], accuracy[1])
