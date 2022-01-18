import sys
sys.path.append('..')
from collections import Counter
import pickle
import numpy as np
from tqdm.auto import tqdm

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
        else: word += l
    return word_list

def wordset(count): #count: 최소 빈도수, vocabulary만들기
    collection = Counter()
    for i in range(100):
        if i < 10: num = '0' + str(i)
        else: num = str(i)
        file = 'data/dataset/news.en-000' + num + '-of-00100'
        words = readtext(file)
        collection += Counter(words)

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

def textfile(word_to_id):
    text = []
    for i in range(100):
        if i < 10 : num = '0' + str(i)
        else: num = str(i)
        file = 'data/dataset/news.en-000' + num + '-of-00100'
        words = readtext(file)

        sentence = []
        for word in words:
            if word in word_to_id.keys():
                sentence.append(word_to_id[word])
                if word == '<EOS>':
                    text.append(np.array(sentence))
                    sentence = []

            else: text.append(word_to_id['<UNK>'])

        #file 저장
        new_file = 'data/dataset/news/en-000' + num + '-of-00100.txt'
        with open(new_file, 'wb') as f:
            pickle.dump(text, f)
        text = []

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

def huffman_encoding():
    with open('./news/vocab.txt', 'rb') as v:
        (word_to_id, id_to_word, count) = pickle.load(v)
        node = tree(count, id_to_word)

    id_to_code, id_to_way = huffman(node)

    max = 0
    for code in id_to_code.values():
        if max < len(code):
            max = len(code)

    l = len(id_to_code)
    mat_code = np.zeros((l, max))
    mat_way = np.zeros((l, max), dtype=int)
    for id in tqdm(range(l)):
        code = id_to_code[id]
        way = id_to_way[id]

        while len(code) < max: code.append(0)
        while len(way) < max: way.append(0)

        mat_code[id] = code
        mat_way[id] = way

    with open('./news/huffman.txt', 'wb') as hf:
        pickle.dump((mat_code, mat_way), hf)
    return None
