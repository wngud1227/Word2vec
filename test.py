import sys
sys.path.append('..')
import numpy as np
from collections import Counter
import pickle
import time

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
    file = './data/news1.txt'
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

def textfile(word_to_id):
    text = []
    file = './data/news1.txt'
    words = readtext(file)
    sentence = []
    for word in words:
        if word in word_to_id.keys():
            sentence.append(word_to_id[word])
            if word == '<EOS>':
                text.append(np.array(sentence))
                sentence = []

        else: sentence.append(word_to_id['<UNK>'])

    #file 저장
    new_file = './data/news1_processed.txt'
    with open(new_file, 'wb') as f:
        pickle.dump(text, f)


# (word_to_id, id_to_word, count) = wordset(5)
# with open('./news/vocab.txt', 'wb') as v:
#     pickle.dump((word_to_id, id_to_word, count), v)

class Node:
    def __init__(self, count, word, left=None, right=None):
        self.count = count
        self.word = word
        self.left = left
        self.right = right

def tree(vocab, id_to_word):
    nodes = []
    for i in range(len(id_to_word)):
        node = Node(vocab[id_to_word[i]], id_to_word[i])
        nodes.append(node)
    while len(nodes) > 1:
        left = nodes[-1]
        right = nodes[-2]
        node = Node(left.count + right.count, None, left, right)
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

def huffman(node, code, codes):
    if(node.left):
        code += 0
        huffman(node.left, code, codes)
    if(node.right):
        code += 1
        huffman(node.right, code, codes)
    else:
        codes[node.word] = code

    return codes

with open('./news/vocab.txt', 'rb') as v:
    (word_to_id, id_to_word, count) = pickle.load(v)
    start = time.time()
    node = tree(count, id_to_word)
    print('tree end : {}'.format((time.time()-start)/60))

code = ''
codes = {}
codes = huffman(node, code, codes)
print(codes)
print('encoding end')

