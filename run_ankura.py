#!/usr/bin/python

import sys
import ankura
import scipy
import os.path
import pickle
import time
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import defaultdict

q_map = {
            'freederp' : ankura.anchor.build_labeled_cooccurrence,
            'supervised' : ankura.anchor.build_supervised_cooccurrence,
            'semi' : ankura.anchor.build_supervised_cooccurrence,
            'vanilla' : ankura.anchor.build_cooccurrence,
            'fclr' : ankura.anchor.build_labeled_cooccurrence, #Run free classifier topics through logistic regression
            'fcdr' : ankura.anchor.build_labeled_cooccurrence,
        }

corpus_map = {
                'tripadvisor' : ankura.corpus.tripadvisor,
                'newsgroups' : ankura.corpus.newsgroups,
                'toy' : ankura.corpus.toy,
                'yelp' : ankura.corpus.yelp,
                'amazon' : ankura.corpus.amazon,
                'newsgroups' : ankura.corpus.newsgroups,
             }

label_map = {
                'newsgroups' : 'coarse_newsgroup',
                'amazon_large' : 'label',
                'tripadvisor' : 'label',
                'yelp' : 'rating',
                'amazon' : 'rating',
                'yelp_binary' : 'binary_rating',
                'tripadvisor_binary' : 'label',
                'amazon_binary' : 'binary_rating',
            }

newsgroup_map = {
                    'politics' : 0,
                    'rec' : 1,
                    'sci' : 2,
                    'religion' : 3,
                    'misc' : 4,
                    'comp' : 5,
                }

binary_map = {
                5.0 : 5.0,
                4.0 : 0.0,
                3.0 : 0.0,
                2.0 : 0.0,
                1.0 : 0.0,
             }

key_map = defaultdict(lambda:identitydict(int))
key_map['newsgroups'] = newsgroup_map

LABEL_NAME = 'label'
Z_ATTR = 'z'

home_dir = os.path.join(os.path.join(os.getenv('HOME'), 'compute'), '.ankura')
PICKLE_FILE = home_dir + '{}results.pickle'

class identitydict(defaultdict):
    def __missing__(self, key):
        return key

def get_logistic_regression_accuracy(train, test, train_target, test_target, topics):

    assign_start = time.time()
    ankura.topic.gensim_assign(train, topics, z_attr=Z_ATTR)
    ankura.topic.gensim_assign(test, topics, z_attr=Z_ATTR)
    assign_end = time.time()

    assign_time = assign_end - assign_start

    matrix_start = time.time()
    train_matrix = scipy.sparse.lil_matrix((len(train.documents), num_topics * len(train.vocabulary)))
    test_matrix = scipy.sparse.lil_matrix((len(test.documents), num_topics * len(test.vocabulary)))

    for i, doc in enumerate(train.documents):
        for j, t in enumerate(doc.tokens):
            train_matrix[i, t[0] * num_topics + doc.metadata[Z_ATTR][j]] += 1

    for i, doc in enumerate(test.documents):
        for j, t in enumerate(doc.tokens):
            test_matrix[i, t[0] * num_topics + doc.metadata[Z_ATTR][j]] += 1

    matrix_end = time.time()
    matrix_time = matrix_end - matrix_start

    print('Running Logistic Regression...')
    lr = LogisticRegression()

    train_start = time.time()
    lr.fit(train_matrix, train_target)
    train_end = time.time()

    train_time = train_end - train_start

    apply_start = time.time()
    predictions = lr.predict(test_matrix)
    apply_end = time.time()

    apply_time = apply_end - apply_start

    count = 0

    for i in range(len(predictions)):
        if predictions[i] == test_target[i]:
            count += 1

    return assign_time, matrix_time, train_time, apply_time, (count / len(predictions))

def free_classifier_dream_accuracy(corpus, test, label, labeled_docs, topics, c, labels):
    classifier = ankura.topic.free_classifier_dream(corpus, label, labeled_docs=labeled_docs, topics=topics, C=c, labels=labels)

    contingency = ankura.validate.Contingency()

    start = time.time()
    for doc in test.documents:
        gold = doc.metadata[label]
        pred = classifier(doc)
        contingency[gold, pred] += 1
    end = time.time()

    apply_time = end - start

    return 0, 0, 0, apply_time, contingency.accuracy()

def get_free_classifier_accuracy(test, topics, Q, labels, label):
    classifier = ankura.topic.free_classifier_derpy(topics, Q, labels)

    print('Getting results from free classifier...')
    start = time.time()
    ankura.topic.variational_assign(test, topics, Z_ATTR)
    contingency = ankura.validate.Contingency()
    for i, doc in enumerate(test.documents):
        contingency[doc.metadata[label], classifier(doc, Z_ATTR)] += 1
    end = time.time()
    apply_time = end - start

    return 0, 0, 0, apply_time, contingency.accuracy()

def run_experiment(corpus_name, model, num_topics=100):

    doc_label_map = key_map[corpus_name]
    label_name = label_map[corpus_name]
    if 'binary' in corpus_name: corpus_name = corpus_name.split('_')[0]

    corpus_retriever = corpus_map[corpus_name]
    q_retriever = q_map[model]


    print('Retrieving corpus...')
    corpus = corpus_retriever()

    total_time_start = time.time()

    print('Splitting corpus into test and train...')
    if model == 'semi' or model == 'freederp' or model == 'fclr' or model == 'fcdr': # Is there a better way to do this?
        split_corpus, test_corpus = ankura.pipeline.test_train_split(corpus, return_ids=True)
        train_corpus, dev_corpus = ankura.pipeline.test_train_split(split_corpus[1], return_ids=True) # Do I care about the dev corpus?
    else:
        train_corpus, test_corpus = ankura.pipeline.test_train_split(corpus, return_ids=True)

    train = train_corpus[1]
    train_labeled_docs = set(train_corpus[0])

    test = test_corpus[1]

    train_target = [doc_label_map[doc.metadata[label_name]] for doc in train.documents]
    test_target = [doc_label_map[doc.metadata[label_name]] for doc in test.documents]

    print('Calculating Q...')
    q_start = time.time()
    if model == 'freederp' or model == 'fclr' or model == 'fcdr':
        Q, labels = q_retriever(split_corpus, label_name, train_labeled_docs)
    elif model == 'supervised':
        Q = q_retriever(train, label_name, train_labeled_docs)
    elif model == 'semi':
        Q = q_retriever(split_corpus, label_name, train_labeled_docs)
    else:
        Q = q_retriever(split_corpus)

    q_end = time.time()
    q_time = q_end - q_start

    print('Retrieving anchors...')
    anchor_start = time.time()
    anchors = ankura.anchor.gram_schmidt_anchors(train, Q, num_topics)
    anchor_end = time.time()
    anchor_time = anchor_end - anchor_start

    print('Retrieving topics...')
    topic_start = time.time()
    c, topics = ankura.anchor.recover_topics(Q, anchors, 1e-5, get_c=True)
    topic_end = time.time()
    
    topic_time = topic_end - topic_start
    
    print('Calculating accuracy...')
    if model == 'freederp':
        assign_time, matrix_time, train_time, apply_time, accuracy = get_free_classifier_accuracy(test, topics, Q, labels, label_name)  
    elif model == 'fcdr':
        assign_time, matrix_time, train_time, apply_time, accuracy = free_classifier_dream_accuracy(split_corpus, test, label_name, train_labeled_docs, topics, c, labels)
    else:
        assign_time, matrix_time, train_time, apply_time, accuracy = get_logistic_regression_accuracy(train, test, train_target, test_target, topics)
    print('Accuracy is:', accuracy)

    summary = ankura.topic.topic_summary(topics)
    coherence = ankura.validate.coherence(corpus, summary)

    total_time_end = time.time()
    total_time = total_time_end - total_time_start

    results = {}
    results['accuracy'] = accuracy
    results['vocab_size'] = len(corpus.vocabulary)
    results['coherence'] = np.sum(coherence) / len(coherence)
    results['topic_time'] = float(topic_time)
    results['anchor_time'] = float(anchor_time)
    results['q_time'] = float(q_time)
    results['total_time'] = float(total_time)
    results['train_time'] = float(train_time)
    results['apply_time'] = float(apply_time)
    results['assign_time'] = float(assign_time)
    results['matrix_time'] = float(matrix_time)

    return results

def create_filtering_directory(filename, run_num, corpus_name, model_name):
    return filename.format('/' + corpus_name + '/' + model_name + '/' + str(num_topics) + '/' + str(run_num))

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print('Invalid number of arguments \n Command: python3 experiments.py <num_topics> <corpus_name> <model> <run_number> <num_iterations>')
        sys.exit()

    num_topics = int(sys.argv[1])
    corpus_name = str(sys.argv[2])
    model = str(sys.argv[3])
    run_number = int(sys.argv[4])
    num_iterations = int(sys.argv[5])

    PICKLE_FILE = create_filtering_directory(PICKLE_FILE, run_number, corpus_name, model)

    results = []
    for num in range(num_iterations):
        results.append(run_experiment(corpus_name, model, num_topics))

    print('Average Accuracy:', np.mean([float(result['accuracy']) for result in results]))
    print('Average Training Time:', np.mean([float(result['train_time']) for result in results]))
    print(results)
    pickle_directory = os.path.dirname(PICKLE_FILE)

    if not os.path.exists(pickle_directory):
        os.makedirs(pickle_directory)

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(results, f)
