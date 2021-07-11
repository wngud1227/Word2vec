import numpy as np
from Preprocess2 import readtext

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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
    test_list = []
    word_list = []
    word = ''
    for l in data:
        if (l == '\n') or (l == ' ') or (l == '\t'):
            word_list.append(word)
            word = ''

            if l == '\n':
                test_list.append(word_list)
                word_list = []

        else: word += l
    return test_list

def accuracy(test_list, W, word_to_id, id_to_word): #input : word_to_id, id_to_word, W, test_list
    correct = 0
    vocab_size = len(id_to_word)

    for test in test_list:
        for i in range(4):
            id = []
            vec = []

            id[i] = word_to_id[test[i]]
            vec[i] = W[id[i]]

        new_vec = vec[0] - vec[1] + vec[2]
        similarity = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity[i] = cosine_similarity(W[i], new_vec)

        new_id = similarity.argsort()[vocab_size - 1]
        if new_id == id[3]:
            correct += 1

    return correct / vocab_size





file = 'data/dataset/news/test.txt'
test_list = make_test(file)
print(test_list)