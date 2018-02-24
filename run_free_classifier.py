#!/usr/bin/python

import sys
import ankura
import sklearn
import random
import bs4
import scipy
import os.path
import pickle
import time
from sklearn.linear_model import LinearRegression
import numpy as np

LABEL_NAME = 'coarse_newsgroup'
Z_ATTR = 'z'
THETA_ATTR = 'theta'

def run_experiment(num_topics=100, label_weight=500, smoothing=1e-4, epsilon=1e-5, train_size=10000, test_size=8000):

    print('Importing corpus...')
    corpus = ankura.corpus.newsgroups()

    total_time_start = time.time()

    print('Splitting training, test sets...')
    train, test = ankura.pipeline.test_train_split(corpus, num_train=train_size, num_test=test_size, return_ids=True)

    print('Constructing Q...')
    Q, labels = ankura.anchor.build_labeled_cooccurrence(corpus, LABEL_NAME, set(train[0]), label_weight, smoothing)

    anchors = ankura.anchor.gram_schmidt_anchors(corpus, Q, num_topics)

    print('Recovering topics...')
    anchor_start = time.time()
    topics = ankura.anchor.recover_topics(Q, anchors, epsilon)
    anchor_end = time.time()
    
    anchor_time = anchor_end - anchor_start

    print('Retrieving free classifier...')
    classifier = ankura.topic.free_classifier(topics, Q, labels)
    ankura.topic.variational_assign(test[1], topics, THETA_ATTR)

    print('Calculating accuracy...')
    contingency = ankura.validate.Contingency()
    for i, doc in enumerate(test[1].documents):
        contingency[doc.metadata[LABEL_NAME], classifier(doc, THETA_ATTR)] += 1

    total_time_end = time.time()
    total_time = total_time_end - total_time_start

    return contingency.accuracy()

if __name__ == "__main__":

    num_topics = int(sys.argv[1])
    label_weight = int(sys.argv[2])
    smoothing = float(sys.argv[3])
    epsilon = float(sys.argv[4])
    train_size = int(sys.argv[5])
    test_size = int(sys.argv[6])

    accuracy = []

    for i in range(3):
        accuracy.append(run_experiment(num_topics, label_weight, smoothing, epsilon, train_size, test_size))

    print(num_topics, label_weight, smoothing, epsilon, np.mean(accuracy))
