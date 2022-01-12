import sys

sys.path.append('..')
import numpy as np
from collections import Counter
import pickle
import time


def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out


def readtext(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    word_list = []
    word = ''
    for l in data:
        if (l == '\n') or (l == ' ') or (l == '\t'):
            word_list.append(word)
            word = ''
            if (l == '\n'):
                word_list.append('<EOS>')
        else:
            word += l
    return word_list


def wordset(count):  # count: 최소 빈도수, vocabulary만들기
    file = './data/dataset/news/news1.txt'
    words = readtext(file)
    collection = Counter(words)

    vocab = {'<UNK>': 0}

    for (word, value) in collection.most_common():
        if value < 5:
            vocab['<UNK>'] += 1
        else:
            vocab[word] = value

    word_to_id = {}
    id_to_word = {}

    for word in vocab.keys():
        word_to_id[word] = len(word_to_id)
        id_to_word[len(id_to_word)] = word

    return word_to_id, id_to_word, vocab


def textfile(word_to_id, new_file='./data/dataset/news/news1_preprocessed.txt'):
    text = []
    file = './data/dataset/news/news1.txt'
    words = readtext(file)
    sentence = []
    for word in words:
        if word in word_to_id.keys():
            sentence.append(word_to_id[word])
            if word == '<EOS>':
                text.append(np.array(sentence))
                sentence = []

        else:
            sentence.append(word_to_id['<UNK>'])

    # file 저장
    with open(new_file, 'wb') as f:
        pickle.dump(text, f)


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


# (word_to_id, id_to_word, count) = wordset(5)
# textfile(word_to_id)
# with open('./data/dataset/vocab_news.txt', 'wb') as v:
#     pickle.dump((word_to_id, id_to_word, count), v)

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
    symbol = 0
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


with open('./data/dataset/vocab_news.txt', 'rb') as v:
    (word_to_id, id_to_word, count) = pickle.load(v)
    start = time.time()
    node = tree(count, id_to_word)
    print('tree end : {}'.format((time.time() - start) / 60))
#
id_to_code, id_to_way = huffman(node)

# max = 0
# for code in id_to_code.values():
#     if max < len(code):
#         max = len(code)
#
# l = len(id_to_code)
#
# mat_code = np.zeros((l, max))
# mat_way = np.zeros((l, max))
# for id in id_to_code.keys():
#     mat_code[id] = id_to_code[id]
#     mat_way[id] = id_to_way[id]
#
# print(mat_way)
# print(mat_code)

# max = 0
# for code in id_to_code.values():
#     if max < len(code):
#         max = len(code)
#
#
# l = len(id_to_code)
# mat_code = np.zeros((l, max))
# mat_way = np.zeros((l, max))
# for id in range(l):
#     code = id_to_code[id]
#     way = id_to_way[id]
#
#     while len(code) < max: code.append(0)
#     while len(way) < max: way.append(0)
#
#     mat_code[id] = code
#     mat_way[id] = way

with open('./data/huffman.txt', 'wb') as hf:
    pickle.dump((id_to_code, id_to_way), hf)
    print('encoding end : {}'.format((time.time() - start) / 60))

# with open('./data/huffman.txt', 'rb') as hf:
#     (id_to_code, id_to_way) = pickle.load(hf)
# with open('./data/news1_processed.txt', 'rb') as data:
#     text = pickle.load(data)
#
#     W_embedding = np.random.uniform(low=-0.5, high=0.5, size=(len(id_to_code), 10))
#     W_embedding_b = np.random.uniform(low=-0.5, high=0.5, size=(len(id_to_code) - 1, 10))
#
#     for sentence in text:
#         contexts, center = create_contexts_target(sentence)
#         for i in range(len(center)):
#             loss = 0
#             for context in contexts[i]:
#                 way = id_to_way[context]
#                 code = id_to_code[context]
#                 v= 1
#                 for j in range(len(way)):
#                     score = code[j] * np.dot(W_embedding_b[way[j]], W_embedding[center[i]])
#
#

#
#
# loss = 0
# for context in contexts:
#     x = W_embedding[context]
#     x = x.reshape(1, 10)
#     for
