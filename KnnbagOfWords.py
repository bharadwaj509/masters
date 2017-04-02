
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import csv
import glob
import pandas as pd
import json
import csv

path = r'annotations'  # use your path
allFiles = glob.glob(path + "/*.csv")

# action first party
total_data_samples0 = []
total_data_number0 = []
total_data_label0 = []

# Personal Info Type
total_data_samples1 = []
total_data_number1 = []
total_data_label1 = []

# Purpose
total_data_samples2 = []
total_data_number2 = []
total_data_label2 = []

# None
total_data_samples3 = []
total_data_number3 = []
total_data_label3 = []

cnt = 0
cnt0 = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
choice = 0
allFiles = allFiles[:85]
for file in allFiles:
    dictCollection = {"Action First-Party": [],
                      "Purpose": [],
                      "Personal Information Type": [],
                      "None": []
                      }

    df = pd.read_csv(file, thousands=',', header=None)
    len(df)
    # df_tail = df.tail(1)[4]
    # print(df_tail)
    number_of_segments = len(df) + 1

    file = file.split('annotations/', 1)[1]
    #     print(file)
    for i in range(number_of_segments - 1):
        # print(i)
        choice = 0  # whether the sentence is from one of the following categories
        if df[5][i] == "First Party Collection/Use":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Action First-Party"]["endIndexInSegment"] != -1:
                total_data_samples0.append(parse_json["Action First-Party"]["selectedText"])
                total_data_number0.append(0)
                total_data_label0.append("Action First-Party")
                cnt0 = cnt0 + 1
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                if parse_json["Purpose"]["endIndexInSegment"] != -1:
                    total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                    total_data_number2.append(2)
                    total_data_label2.append("Purpose")
                    cnt2 = cnt2 + 1
        if df[5][i] == "Third Party Sharing/Collection":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                total_data_number2.append(2)
                total_data_label2.append("Purpose")
                cnt2 = cnt2 + 1
                cnt1 = cnt1 + 1
                total_data_label1.append("Personal Information Type")
                total_data_number1.append(1)
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
        if df[5][i] == "User Choice/Control":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                total_data_number2.append(2)
                total_data_label2.append("Purpose")
                cnt2 = cnt2 + 1
        if df[5][i] == "Data Retention":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
        if choice == 0:
            #             print("practis is -->", df[5][i])
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            #             print(len(df[6][i]))
            #             print(parse_json.keys())
            attributes = parse_json.keys()
            #             print(attributes)
            for k in attributes:
                #                 print(k)
                if parse_json[k]['startIndexInSegment'] != -1:
                    #                     print(parse_json[k]['selectedText'])
                    total_data_samples3.append(parse_json[k]['selectedText'])
                    total_data_number3.append(3)
                    total_data_label3.append("None")
                    cnt3 = cnt3 + 1

print("---")
print(cnt0)
print(cnt1)
print(cnt2)
print(cnt3)
print("--")
print(cnt0 + cnt1 + cnt2 + cnt3)
print(len(total_data_samples0))
print(len(total_data_samples1))
print(len(total_data_samples2))
print(len(total_data_samples3))

data_samples1 = total_data_samples0[:4000]
data_samples1.extend(total_data_samples1[0:4000])
data_samples1.extend(total_data_samples2[0:4000])
data_samples1.extend(total_data_samples3[0:4000])
print()
print(len(data_samples1))

data_number1 = total_data_number0[:4000]
data_number1.extend(total_data_number1[:4000])
data_number1.extend(total_data_number2[:4000])
data_number1.extend(total_data_number3[:4000])
print()
print(len(data_number1))

data_label1 = total_data_label0[:4000]
data_label1.extend(total_data_label1[:4000])
data_label1.extend(total_data_label2[:4000])
data_label1.extend(total_data_label3[:4000])
print()
print(len(data_label1))

test_data_samples1 = total_data_samples0[4001:5200]
test_data_samples1.extend(total_data_samples1[4001:5200])
test_data_samples1.extend(total_data_samples2[4001:5200])
test_data_samples1.extend(total_data_samples3[4001:5200])
print()
print(len(test_data_samples1))

test_data_number1 = total_data_number0[4001:5200]
test_data_number1.extend(total_data_number1[4001:5200])
test_data_number1.extend(total_data_number2[4001:5200])
test_data_number1.extend(total_data_number3[4001:5200])
print()
print(len(test_data_number1))

test_data_label1 = total_data_label0[4001:5200]
test_data_label1.extend(total_data_label1[4001:5200])
test_data_label1.extend(total_data_label2[4001:5200])
test_data_label1.extend(total_data_label3[4001:5200])
print()
print(len(test_data_label1))

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
              " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load some categories from the training set
categories = [
              'Action First-Party',
              'Purpose',
              'Personal Information Type'
              ]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading privacy policies dataset for categories:")
# print(data_label1)

print("%d documents" % len(data_samples1))
print("%d categories" % len(data_label1))
print()

labels = data_number1
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

print("fdsafdaf4")
vectorizer = TfidfVectorizer(max_df=0.5, max_features=20000,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(data_samples1)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

