import sys

sys.path.append('..')
import numpy as np
from collections import Counter
import pickle
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

class Node:
    def __init__(self, count, id, symbol=None, left=None, right=None):
        self.count = count
        self.id = id
        self.symbol = symbol
        self.left = left
        self.right = right


def tree(vocab, id_to_word):
    nodes = []
    for i in range(len(id_to_word)):
        node = Node(vocab[id_to_word[i]], i)
        nodes.append(node)
    symbol = int(0)
    while len(nodes) > 1:
        left = nodes[-1]
        right = nodes[-2]
        node = Node(left.count + right.count, None, symbol, left, right)
        symbol += 1
        nodes.remove(left)
        nodes.remove(right)
        leng = len(nodes)
        for i in range(len(nodes), 0, -1):
            if nodes[i - 1].count >= node.count:
                nodes.insert(i, node)
                break
        if len(nodes) == leng:
            nodes.insert(0, node)
    return nodes[0]


def huffman(node, code=[], id_to_code={}, id_to_way={}, way=[]):
    if (node.symbol != None):
        if (node.left != None):
            new_way = way.copy()
            new_way.append(node.symbol)
            new_code = code.copy()
            new_code.append(1)
            huffman(node.left, new_code, id_to_code, id_to_way, new_way)
        if (node.right != None):
            new_way = way.copy()
            new_way.append(node.symbol)
            new_code = code.copy()
            new_code.append(-1)
            huffman(node.right, new_code, id_to_code, id_to_way, new_way)

    else:
        id_to_code[node.id] = code
        id_to_way[node.id] = way

    return id_to_code, id_to_way


# with open('./data/vocab.txt', 'rb') as v:
#     (word_to_id, id_to_word, count) = pickle.load(v)
#     start = time.time()
#     node = tree(count, id_to_word)
#     print('tree end : {}'.format((time.time() - start) / 60))
# #
# id_to_code, id_to_way = huffman(node)
#
# max = 0
# for code in id_to_code.values():
#     if max < len(code):
#         max = len(code)
#
# l = len(id_to_code)
# mat_code = np.zeros((l, max))
# mat_way = np.zeros((l, max), dtype=int)
# for id in range(l):
#     code = id_to_code[id]
#     way = id_to_way[id]
#
#     while len(code) < max: code.append(0)
#     while len(way) < max: way.append(0)
#
#     mat_code[id] = code
#     mat_way[id] = way
#
# print(mat_way)
# with open('./data/huffman.txt', 'wb') as hf:
#     pickle.dump((mat_code, mat_way), hf)
#     print('encoding end : {}'.format((time.time() - start) / 60))

#train
with open('./data/vocab.txt', 'rb') as v:
    (word_to_id, id_to_word, count) = pickle.load(v)
with open('./data/huffman.txt', 'rb') as hf:
    (id_to_code, id_to_way) = pickle.load(hf)
with open('./data/news1_processed.txt', 'rb') as data:
    text = pickle.load(data)
    lr = 0.025
    W_embedding = np.random.uniform(low=-0.5, high=0.5, size=(len(id_to_code), 10))
    W_embedding_b = np.random.uniform(low=-0.5, high=0.5, size=(len(id_to_code) - 1, 10))

    n=0
    total_loss = 0
    for sentence in tqdm(text):
        n += 1
        contexts, center = create_contexts_target(sentence, window_size=3)

        for i in range(len(center)):
            #forward
            code = id_to_code[center[i]]    #(V-1,)
            way = list(id_to_way[center[i]])      #(V-1,)
            out = W_embedding[contexts[i]]  #(window, D)
            score = code * np.dot(out, W_embedding_b[way].T)    #(window, V-1)

            loss = np.sum(-np.log(sigmoid(score) + 1e-07))
            loss /= len(contexts[i])
            total_loss += loss

            #backward
            dout = sigmoid(score)           #(window, V-1)
            dout = code * (dout - 1)        #(window, V-1)
            dx = np.dot(dout, W_embedding_b[way])     #(window, D)
            dW_out = np.dot(dout.T, out)    #(V-1, D)

            W_embedding_b[way] -= dW_out * lr
            W_embedding[contexts[i]] -= dx * lr

        if n%10000==0:
            print(total_loss/n)
            total_loss=0


def cosine_similarity(predict, word_vectors):
    norm_predict = np.linalg.norm(predict, axis=1)
    norm_words = np.linalg.norm(word_vectors, axis=1)

    similarity = np.dot(predict, word_vectors.T)      # similarity = (N, V)
    similarity *= 1/norm_words
    similarity = similarity.T
    similarity *= 1/norm_predict
    similarity = similarity.T

    return similarity

def most_similar(word, W, word_to_id, id_to_word, top=5):
    if word not in word_to_id:
        print("Cannot find the word %s." % word)
        return
    id = word_to_id[word]
    vec = W[id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cosine_similarity(W[i], vec)

    count = 0
    sim_word = []
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == word:
            continue
        print("word: {}, similarity: {}".format(id_to_word[i], similarity[i]))
        sim_word[count] = id_to_word[i]
        count += 1

        if count >= top:
            return sim_word

def make_test(file): #input : test.txt
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()

    semantic_test = []
    syntactic_test = []
    a = 0   #a=0 -> semantic, a=1 -> syntactic
    word_list = []
    word = ''
    for l in data:
        if (l == '\n') or (l == ' ') or (l == '\t'):
            if word == '':
                continue

            word_list.append(word)
            word = ''

            if l == '\n':
                if a == 0:
                    semantic_test.append(word_list)
                elif a == 1:
                    syntactic_test.append(word_list)
                word_list = []

        elif word == 'gram':
            a = 1
        else: word += l

    return semantic_test, syntactic_test

def save_test(file, word_to_id):
    text = make_test(file)
    semantic_test = []
    syntactic_test = []
    test = []
    for t in text[0]:
        if len(t) != 4:
            continue
        for word in t:
            if word in word_to_id.keys():
                test.append(word_to_id[word])

            else:
                test = []
                break

        if len(test) == 4:
            semantic_test.extend(test)
            test = []

    for t in text[1]:
        if len(t) != 4:
            continue
        for word in t:
            if word in word_to_id.keys():
                test.append(word_to_id[word])

            else:
                test = []
                break

        if len(test) == 4:
            syntactic_test.extend(test)
            test = []

    semantic_test = np.array(semantic_test).reshape([-1, 4])
    syntactic_test = np.array(syntactic_test).reshape([-1, 4])

    # file 저장
    new_file = './data/test_labeled.pkl'
    with open(new_file, 'wb') as f:
        pickle.dump((semantic_test, syntactic_test), f)
    return None

def embedding(test_list, W):
    a = []
    b = []
    c = []
    d = []
    for i in test_list:
        a_temp = W[i[0]]
        b_temp = W[i[1]]
        c_temp = W[i[2]]
        d_temp = W[i[3]]

        a_norm = np.linalg.norm(a_temp)
        b_norm = np.linalg.norm(b_temp)
        c_norm = np.linalg.norm(c_temp)
        d_norm = np.linalg.norm(d_temp)

        a.append(a_temp / a_norm)
        b.append(b_temp / b_norm)
        c.append(c_temp / c_norm)
        d.append(d_temp / d_norm)

    return np.array(a), np.array(b), np.array(c), np.array(d)

def accuracy(file, W): #input : word_to_id, id_to_word, W, test_list
    with open(file, 'rb') as f:
        semantic, syntactic = pickle.load(f)

    correct = 0
    batch_size = len(semantic) // 10 + 1
    for i in tqdm(range(10), desc='semantic: '):
        test_batch = semantic[i*batch_size:(i+1)*batch_size]
        a, b, c, d = embedding(test_batch, W)
        new_vec = b - a + c     #new_vec = (test_batch, dimension)
        similarity = cosine_similarity(new_vec, W)
        new_id = similarity.argsort(axis=1)
        new_id = new_id[:, ::-1]
        new_id = new_id[:, :4]
        for j in range(len(new_id)):
            if test_batch[j, 3] == new_id[j][0]:
                correct += 1
    semantic_accuracy = correct / len(semantic)

    correct = 0
    batch_size = len(syntactic) // 10 + 1
    for i in tqdm(range(10), desc='syntactic: '):
        test_batch = syntactic[i*batch_size:(i+1)*batch_size]
        a, b, c, d = embedding(test_batch, W)
        new_vec = b - a + c     #new_vec = (test_batch, dimension)
        similarity = cosine_similarity(new_vec, W)
        new_id = similarity.argsort(axis=1)
        new_id = new_id[:, ::-1]
        new_id = new_id[:, :4]
        for j in range(len(new_id)):
            if test_batch[j, 3] == new_id[j][0]:
                correct += 1
    syntactic_accuracy = correct / len(syntactic)

    return semantic_accuracy, syntactic_accuracy

#test
# save_test(file='./data/test.txt', word_to_id=word_to_id)
accuracy = accuracy(file='./data/test_labeled.pkl', W=W_embedding)
print(accuracy[0], accuracy[1])
