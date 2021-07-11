import sys
sys.path.append('..')
from collections import Counter
import pickle
import numpy as np

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


# def huffman(vocab, counts): #vocab: 단어 목록, counts: 단어 count
#     num = len(vocab)
#     pos1 = num
#     pos2 = num - 1



