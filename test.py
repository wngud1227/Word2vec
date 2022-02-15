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
    def __init__(self, counts, word_to_id, t=1e-05):
        value = list(counts.values())
        self.whole_vocab = np.sum(value)
        self.del_p = {}
        for (word, f) in counts.items():
            temp = f / self.whole_vocab
            id = word_to_id[word]
            self.del_p[id] = (1 + np.sqrt(temp / t)) * (t / temp)

    def delete_vocab(self, sentence):
        new_sentence = []
        for word in sentence:
            if self.del_p[word] < np.random.rand():
                continue
            else:
                new_sentence.append(word)
        return new_sentence

class UnigramTable:
    def __init__(self, vocab, word_to_id, power=0.75):
        self.table = []
        for word in vocab.keys():
            id = word_to_id[word]
            if id == 0: continue
            count = int(vocab[word] ** power)
            self.table.extend([id] * count)
        self.p_sum = len(self.table)
        self.table = np.array(self.table)

    def negative_sampling(self, target, num_sample=5):
        while(1):
            index = np.random.randint(low=0, high=self.p_sum, size=num_sample)
            neg_sample = self.table[index]
            if target in neg_sample:
                continue
            else:
                neg = np.append([target], neg_sample)
                return neg

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
        self.subsampling = subsampling(self.vocab, self.word_to_id)
        self.Unigram = UnigramTable(self.vocab, self.word_to_id)
        self.W_embedding = 0.01*np.random.randn(self.vocab_size, self.dimension).astype('f')
        self.W_embedding_b = np.zeros((self.vocab_size, self.dimension)).astype('f')

    def forward(self, contexts, target):
        loss = 0
        cache = []
        for context in contexts:
            x = self.W_embedding[context].reshape(1, -1)
            negative_sample = self.Unigram.negative_sampling(target, num_sample=5)
            out = sigmoid(np.dot(x, self.W_embedding_b[negative_sample].T))
            p_loss = -np.log(out[:, :1] + 1e-07)
            n_loss = -np.sum(np.log(1 - out[:, 1:] + 1e-07))
            loss += (p_loss + n_loss)
            cache.append((x, context, negative_sample, out))

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
                    max_n = len(text)
                for sentence in text:
                    n += 1
                    #subsampling
                    new_sentence = self.subsampling.delete_vocab(sentence=sentence)
                    # contexts, target = create_contexts_target(sentence, window_size=self.window_size + 1)
                    contexts, target = create_contexts_target(new_sentence, window_size=self.window_size + 1)


                    for i in range(len(target)):
                        # lr decay
                        alpha = 1 - n / (self.num_sentence * epoch)
                        if alpha <= 0.0001:
                            alpha = 0.0001
                        lr = self.lr * alpha

                        #train
                        count += 1
                        loss += self.forward(contexts[i], target[i])
                        self.backward(lr)
                        if count % 10000 == 0:
                            train_time = (time.time() - start) / 3600
                            avg_loss = loss/count
                            print('time: {}, loss : {}'.format(train_time, avg_loss))
                            print('{}/{} sentence trained!'.format(n, max_n))
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

#     test = make_test(file)
#     save_test(file, word_to_id)
#
# with open('data/embedding_sg_neg5.pkl', 'rb') as f:
#     W_embedding = pickle.load(f)
# with open('news/vocab.txt', 'rb') as v:
    # (word_to_id, id_to_word, vocab) = pickle.load(v)
# most_similar(word='king', W=model.W_embedding, word_to_id=word_to_id, id_to_word=id_to_word)
# accuracy = accuracy(file='./news/test_labeled.pkl', W=W_embedding)
# print(accuracy)
# (0.07812908283250589, 0.1338573575309213, 0.11676550729283539) 1iter
