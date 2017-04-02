
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




path = r'Train'  # use your path
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

print(len(allFiles))


for file in allFiles:
    # print(file)
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

    file = file.split('Train/', 1)[1]
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
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
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

data_samples1 = total_data_samples0[:3673]
data_samples1.extend(total_data_samples1[0:3673])
data_samples1.extend(total_data_samples2[0:3673])
data_samples1.extend(total_data_samples3[0:3673])
print()
print(len(data_samples1))

data_number1 = total_data_number0[:3673]
data_number1.extend(total_data_number1[:3673])
data_number1.extend(total_data_number2[:3673])
data_number1.extend(total_data_number3[:3673])
print()
print(len(data_number1))

data_label1 = total_data_label0[:3673]
data_label1.extend(total_data_label1[:3673])
data_label1.extend(total_data_label2[:3673])
data_label1.extend(total_data_label3[:3673])
print()
print(len(data_label1))






path = r'Test'  # use your path
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

print(len(allFiles))


for file in allFiles:
    # print(file)
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

    file = file.split('Test/', 1)[1]
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
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
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



test_data_samples1 = total_data_samples0[0:1208]
test_data_samples1.extend(total_data_samples1[0:1208])
test_data_samples1.extend(total_data_samples2[0:1208])
test_data_samples1.extend(total_data_samples3[0:1208])
print()
print(len(test_data_samples1))

test_data_number1 = total_data_number0[0:1208]
test_data_number1.extend(total_data_number1[0:1208])
test_data_number1.extend(total_data_number2[0:1208])
test_data_number1.extend(total_data_number3[0:1208])
print()
print(len(test_data_number1))

test_data_label1 = total_data_label0[0:1208]
test_data_label1.extend(total_data_label1[0:1208])
test_data_label1.extend(total_data_label2[0:1208])
test_data_label1.extend(total_data_label3[0:1208])
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

# print("%d documents" % len(data_samples1))
# print("%d categories" % len(data_label1))
# print()

labels = data_number1
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

# print("fdsafdaf4")
vectorizer = TfidfVectorizer(max_df=0.5, max_features=20000,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(data_samples1)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


#
###############################################################################
# Do the actual clustering

# if opts.minibatch:
#     km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
#                          init_size=1000, batch_size=1000, verbose=opts.verbose)
# else:
km = KMeans(n_clusters=10, init='k-means++', max_iter=20000, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)

print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
# print(labels)
print(np.unique(km.labels_))
print(true_k)
print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(10):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()




Y = vectorizer.fit_transform(test_data_samples1)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % Y.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    Y = lsa.fit_transform(Y)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


km_pre = km.predict(Y)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km_pre.labels_))
print(km_pre[:10])

