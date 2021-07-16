model = CBOW(window_size=3, hidden_unit=500)
model.train(epoch=1)


#ERROR CODE
# Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
# Iteration:   0%|          | 0/100 [00:00<?, ?it/s]1 sentence
# 24.94971286342395
# 2 sentence
# 12.475686434077222
# 3 sentence
# 74.85785025569916
# 4 sentence
# 108.12904407989319
# 5 sentence
# 29.11060249387114
# 6 sentence
# 91.49241108015433
# 7 sentence
# 33.270659758847295
# 8 sentence
# 0
# 9 sentence
# 12.47591703306918
# 10 sentence
# 137.23573788791634
# Iteration:   0%|          | 0/100 [02:56<?, ?it/s]
# Epoch:   0%|          | 0/1 [02:56<?, ?it/s]
# Traceback (most recent call last):
#   File "C:/Users/박주형/PycharmProjects/Word2vec/test.py", line 151, in <module>
#     model.train(epoch=1)
#   File "C:\Users\박주형\PycharmProjects\Word2vec\word2vec2.py", line 257, in train
#     contexts, target = create_contexts_target(sentence, window_size=self.window_size + 1)
#   File "C:\Users\박주형\PycharmProjects\Word2vec\word2vec2.py", line 41, in create_contexts_target
#     target = corpus[window_size:-window_size]
# TypeError: 'int' object is not subscriptable

# Process finished with exit code 1
