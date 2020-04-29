import os
import sys
from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def preprocess(train_sents, test_sents):
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    print("===Data===")
    print(X_train[0])
    print(y_train[0])
    print("======")
    return (X_train, y_train), (X_test, y_test)


def find_best_params(X_train, y_train, labels):
    "运行时间较长"
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=70,
        all_possible_transitions=True)
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),}
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)
    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=30,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)
    print("\n\n===CV===")
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    print("======")
    return rs.best_params_


def train(train_sents, test_sents):
    from itertools import chain
    (X_train, y_train), (X_test, y_test) = preprocess(train_sents, test_sents)
    labels = list(set(chain.from_iterable(y_train)))
    labels.remove('O')
    params = find_best_params(X_train[:3000], y_train[:3000], labels)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=params["c1"],
        c2=params["c2"],
        max_iterations=100,
        all_possible_transitions=True)
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0]))
    print("\n\n===train===")
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3))
    print("======")
    return crf


def show_results(crf):
    from collections import Counter

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("\n\n===Result===")

    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])


if __name__ == "__main__":
    from nltk.corpus import conll2002

    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    crf = train(train_sents, test_sents)
    show_results(crf)
