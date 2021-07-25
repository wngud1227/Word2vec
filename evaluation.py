import numpy as np
import pickle
from tqdm.auto import tqdm
from Preprocess2 import readtext

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
    new_file = 'data/dataset/news/test_labeled.pkl'
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


    # vec = np.zeros([test_list.shape[0], 300])
    # for i in range(4):
    #     vec.append(W[id[i]])
    #
    # new_vec = vec[0] - vec[1] + vec[2]
    # similarity = np.zeros(vocab_size)
    # for i in range(vocab_size):
    #     similarity[i] = cosine_similarity(W[i], new_vec)
    #
    # new_id = similarity.argsort()[vocab_size - 1]
    # if new_id == id[3]:
    #     correct += 1
    #
    # return correct / test_list.shape[0]
