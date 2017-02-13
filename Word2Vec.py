# importing the extracted documents and performing word-vectorization
from __future__ import absolute_import, division, print_function
import glob
import pandas as pd
import json
import csv
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = r'annotations'  # use your path
allFiles = glob.glob(path + "/*.csv")

raw_sentences = []

for file in allFiles:

    dictCollection = {"Action First-Party":[],
                      "Purpose":[],
                      "Personal Information Type":[]
                      }

    df = pd.read_csv(file, thousands=',', header=None)
    len(df)
    # df_tail = df.tail(1)[4]
    # print(df_tail)
    number_of_segments = len(df) + 1

    file = file.split('annotations/', 1)[1]

    myFile = open('collections/' + file, 'a', newline='')
    wr = csv.writer(myFile)
    # print(file)
    # print(number_of_segments)
    for i in range(number_of_segments-1):
        # print(i)
        if df[5][i] == "First Party Collection/Use":
            parse_json = json.loads(str(df[6][i]))
            if parse_json["Action First-Party"]["endIndexInSegment"] != -1:
                raw_sentences.append(parse_json["Action First-Party"]["selectedText"])
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                raw_sentences.append(parse_json["Personal Information Type"]["selectedText"])
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                raw_sentences.append(parse_json["Purpose"]["selectedText"])

print(raw_sentences)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1

policy2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

policy2vec.build_vocab(sentences)


if not os.path.exists("trained"):
    os.makedirs("trained")

policy2vec.save(os.path.join("trained", "policy2vec.w2v"))

policy2vec = w2v.Word2Vec.load(os.path.join("trained", "policy2vec.w2v"))


policy2vec.most_similar("demographics")


# from string import translate, maketrans, punctuation
# from itertools import chain
# from nltk import PunktSentenceTokenizer
# import datetime
# import re
#
#
# def log(msg):
#     print("{} {}".format(str(datetime.datetime.now()), msg))
#
#
# def removeNonAscii(s):
#     return "".join(filter(lambda x: ord(x) < 128, s))
#
#
# # keeps -, +, # in words
# punctuation = punctuation.replace('-', '').replace('+', '').replace('#', '')
# # makes a C translation dictionary converting punctuations to white spaces
# Trans = maketrans(punctuation, ' ' * len(punctuation))
# # splits text into sentences'
# tknr = PunktSentenceTokenizer()
#
#
# # fast ngrammer if you end up using it for phrases
# def ngrammer2(l, n):
#     temp = [" ".join(l[i:i + n]) for i in xrange(0, len(l)) if len(l[i:i + n]) == n]
#     return temp
#
#
# print 'Loading the post data'
# import pickle
#
# s = pickle.load(open("title_and_job.p", "rb"))
# x_train_RAW = []
# for i in s:
#     if len(i.values()[0]) >= 30:
#         title = i.keys()[0]
#         for q in i.values()[0]:
#             x_train_RAW.append(q.encode('utf-8'))
#
#
# # can use the ngrammer here if you want to look at phrase similarity
# # I get rid of html characters from this corpus
# def spliter(jobpost):
#     sentences2 = []
#     s = tknr.tokenize(jobpost)
#     cleaned_words = [list(translate(
#         re.sub(r'[0-9]|\-|\\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\[|\]|\\|\<|\<|\>|\?|\/|\;|\\.', ' ',
#                sentence).lower().encode('utf-8'), Trans).split()) for sentence in s]
#     # two_three_ngrams = [ngrammer2(sent,num) for num in [1,2,3] for sent in cleaned_words]
#     for U in cleaned_words:
#         sentences2.append(U)
#     sentences2 = list(chain(*sentences2))
#     return sentences2
#
#
# # i always do this, not sure why.
# import random
#
# random.shuffle(x_train_RAW)
#
# # going to multiprocess the tokenizer to make it faster
# from multiprocessing import Pool, cpu_count
#
# pool = Pool(cpu_count())
# print 'starting to sentence tokenize'
# x_train_RAW = filter(None, pool.map(spliter, x_train_RAW))
#
# import gensim
# from multiprocessing import cpu_count
#
# model = gensim.models.Word2Vec(x_train_RAW, size=100, window=5, min_count=5, workers=cpu_count())
# pickle.dump(model, open('model.p', 'wb'))
