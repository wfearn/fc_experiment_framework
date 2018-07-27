#!/usr/bin/python

import sys
import signal
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
                'ammod' : ankura.corpus.amazon_modified,
                'amlrg' : ankura.corpus.amazon_large,
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
                'ammod' : 'label',
                'amlrg' : 'label',
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
                True : 1,
                False : 0,
             }

float_binary_map = {
                        1.0 : 0.0,
                        2.0 : 0.0,
                        3.0 : 0.0,
                        4.0 : 0.0,
                        5.0 : 1.0,
                   }

key_map = defaultdict(lambda:identitydict(int))
key_map['newsgroups'] = newsgroup_map
key_map['amazon_binary'] = binary_map
key_map['yelp_binary'] = binary_map
key_map['ammod'] = float_binary_map
key_map['amlrg'] = float_binary_map

LABEL_NAME = 'label'
THETA_ATTR = 'z'

HOMEDIR = os.path.join(os.getenv('HOME'), '.ankura')
PICKLE_FILE = f'{HOMEDIR}{{}}results.pickle'
LOCALDIR = '/local/amazon_large'

class identitydict(defaultdict):
    def __missing__(self, key):
        return key

def get_logistic_regression_accuracy(train, test, train_target, test_target, topics, label):

    assign_start = time.time()
    train_thetas = ankura.topic.gensim_assign(train, topics, theta_attr=THETA_ATTR)
    test_thetas = ankura.topic.gensim_assign(test, topics, theta_attr=THETA_ATTR)
    assign_end = time.time()

    assign_time = assign_end - assign_start

    matrix_start = time.time()
    train_matrix = np.zeros((len(train.documents), num_topics))
    test_matrix = np.zeros((len(test.documents), num_topics))

    def log_doc_topic(i, thetas):
        return np.log(thetas[i] + 1e-30)

    for i, doc in enumerate(train.documents):
        train_matrix[i, :] = log_doc_topic(i, train_thetas)

    for i, doc in enumerate(test.documents):
        test_matrix[i, :] = log_doc_topic(i, test_thetas)

    matrix_end = time.time()
    matrix_time = matrix_end - matrix_start

    print('Running Logistic Regression...')
    sys.stdout.flush()
    lr = LogisticRegression()

    train_start = time.time()
    lr.fit(train_matrix, train_target)
    train_end = time.time()

    train_time = train_end - train_start

    apply_start = time.time()
    accuracy = lr.score(test_matrix, test_target)
    apply_end = time.time()

    apply_time = apply_end - apply_start

    return assign_time, matrix_time, train_time, apply_time, accuracy


def run_experiment(corpus_name='amlrg', model='semi', num_topics=80, seed=None, vocab_size=3000, run_number=97):

    total_time_start = time.time()

    doc_label_map = key_map[corpus_name]
    label_name = label_map[corpus_name]

    corpus_retriever = corpus_map[corpus_name]
    q_retriever = q_map[model]

    print('Retrieving corpus...')
    sys.stdout.flush()
    docs_path, corpus_path, corpus = corpus_retriever(hash_size=vocab_size)

    print('Corpus Path:', corpus_path)
    print('Docs Path:', docs_path)
    print('Vocabulary:', len(corpus.vocabulary))
    print('Splitting corpus into test and train...')
    sys.stdout.flush()


    split_corpus, test_corpus = ankura.pipeline.train_test_split(corpus, random_seed=seed, return_ids=True, save_dir=LOCALDIR, vocab_size=vocab_size, train_name='split')

    train_corpus, dev_corpus = ankura.pipeline.train_test_split(split_corpus[1], random_seed=seed, return_ids=True, save_dir=LOCALDIR, vocab_size=vocab_size, test_name='dev')

    split = split_corpus[1]
    train = train_corpus[1]
    test = test_corpus[1]
    split_ids = split_corpus[0]
    train_labeled_docs = set([split_ids[i] for i in train_corpus[0]])

    train_target = [doc_label_map[doc.metadata[label_name]] for doc in train.documents]
    test_target = [doc_label_map[doc.metadata[label_name]] for doc in test.documents]

    print('Calculating Q...')
    sys.stdout.flush()
    q_start = time.time()
    Q = q_retriever(split, label_name, train_labeled_docs)
    q_end = time.time()
    q_time = q_end - q_start

    print('Retrieving anchors...')
    sys.stdout.flush()

    anchor_start = time.time()
    anchors = ankura.anchor.gram_schmidt_anchors(train, Q, num_topics)
    anchor_end = time.time()
    anchor_time = anchor_end - anchor_start

    print('Retrieving topics...')
    sys.stdout.flush()

    topic_start = time.time()
    c, topics = ankura.anchor.recover_topics(Q, anchors, 1e-5, get_c=True)
    topic_end = time.time()

    topic_time = topic_end - topic_start

    print('Calculating accuracy...')
    sys.stdout.flush()

    assign_time, matrix_time, train_time, apply_time, accuracy = get_logistic_regression_accuracy(train, test, train_target, test_target, topics, label_name)

    print('Accuracy is:', accuracy)
    sys.stdout.flush()

    total_time_end = time.time()
    total_time = total_time_end - total_time_start

    summary = ankura.topic.topic_summary(topics)
    coherence = ankura.validate.coherence(corpus, summary)

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
    results['num_topics'] = num_topics
    results['seed'] = seed
    results['corpus'] = corpus_name
    results['train_size'] = len(train.documents)
    results['test_size'] = len(test.documents)

    os.remove(docs_path)
    os.remove(corpus_path)

    return results

def create_filtering_directory(filename, run_num, corpus_name, model_name):
    return filename.format('/' + corpus_name + '/' + model_name + '/' + str(num_topics) + '/' + str(run_num))

def sigterm_handler(signum, stack_frame):
    os.remove(docs_path)
    os.remove(corpus_path)

    sys.exit(0)

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print('Invalid number of arguments \n Command: python3 experiments.py <num_topics> <corpus_name> <model> <run_number> <num_iterations>')
        sys.exit()

    num_topics = int(sys.argv[1])
    corpus_name = str(sys.argv[2])
    model = str(sys.argv[3])
    run_number = int(sys.argv[4])
    num_iterations = int(sys.argv[5])
    vocab_size = int(sys.argv[6])
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else None

    print('Topics:', num_topics)
    print('Corpus:', corpus_name)
    print('Model:', model)
    print('Run Num:', run_number)
    print('Num Iterations:', num_iterations)
    print('Seed:', seed)
    print('Vocabulary Size:', vocab_size)

    signal.signal(signal.SIGTERM, sigterm_handler)

    PICKLE_FILE = create_filtering_directory(PICKLE_FILE, run_number, corpus_name, model)
    print('Pickle File:', PICKLE_FILE)

    results = []
    for num in range(num_iterations):
        results.append(run_experiment(corpus_name=corpus_name, model=model, num_topics=num_topics, vocab_size=vocab_size, run_number=run_number))

    print('Average Accuracy:', np.mean([float(result['accuracy']) for result in results]))
    print('Average Training Time:', np.mean([float(result['train_time']) for result in results]))
    print('Average Total Time:', np.mean([float(result['total_time']) for result in results]))
    print('Average Train Time:', np.mean([float(result['train_time']) for result in results]))
    print('Average Apply Time:', np.mean([float(result['apply_time']) for result in results]))
    print(results)
    pickle_directory = os.path.dirname(PICKLE_FILE)

    if not os.path.exists(pickle_directory):
        os.makedirs(pickle_directory)

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(results, f)
