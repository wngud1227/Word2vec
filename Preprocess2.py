import sys
sys.path.append('..')
from collections import Counter
import pickle
import numpy as np
import heapq
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

    def __lt__(self, other):
        return self.count < other.count

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, Node)):
            return False
        return self.count == other.count


def make_node(id_to_word, vocab):
    nodes = []
    for id in tqdm(id_to_word.keys()):
        node = Node(vocab[id_to_word[id]], id)
        heapq.heappush(nodes, node)

    symbol = int(0)
    start = time.time()
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        node = Node(left.count + right.count, None, symbol, left, right)
        symbol += 1
        heapq.heappush(nodes, node)

        if symbol%10000 == 0:
            print(time.time() - start)
    print('Make Node End!')
    return nodes


def make_codes(root, current_code, current_way, codes, way):
    if (root == None):
        return

    if (root.id != None):
        codes[root.id] = current_code
        way[root.id] = current_way
        return

    if root.left:
        new_code, new_way = current_code.copy(), current_way.copy()
        new_code.append(-1)
        new_way.append(root.symbol)
        make_codes(root.left, new_code, new_way, codes, way)
    if root.right:
        new_code, new_way = current_code.copy(), current_way.copy()
        new_code.append(1)
        new_way.append(root.symbol)
        make_codes(root.right, new_code, new_way, codes, way)

def huffman(nodes):
    codes, way = {}, {}
    current_code, current_way = [], []
    root = heapq.heappop(nodes)
    make_codes(root, current_code, current_way, codes, way)
    print('Make Code End!')

    with open('./data/huffman_new.txt', 'wb') as hf:
        pickle.dump((codes, way), hf)

    return
