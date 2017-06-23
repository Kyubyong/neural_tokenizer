# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neural_tokenizer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import re


def load_vocab():
    vocab = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'-" # _: sentinel for Padding
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_data(mode="train"):
    word2idx, idx2word = load_vocab()

    from nltk.corpus import brown
    sents = [" ".join(words) for words in brown.sents()]

    xs, ys = [], []
    for sent in sents:
        sent = re.sub(r"[^ A-Za-z']", "", sent)
        if hp.minlen <= len(sent) <= hp.maxlen:
            x, y = [], []
            for word in sent.split():
                for char in word:
                    x.append(word2idx[char])
                    y.append(0) # 0: no space
                y[-1] = 1 # space for end of a word
            y[-1] = 0 # no space for end of sentence

            xs.append(x + [0] * (hp.maxlen-len(x)))
            ys.append(y + [0] * (hp.maxlen-len(x)))

    # Convert to ndarrays
    X = np.array(xs, np.int32)
    Y = np.array(ys, np.int32)

    # mode
    if mode=="train":
        X, Y = X[: int(len(X) * .8)], Y[: int(len(Y) * .8)]
        # X, Y = X[: 128], Y[: 128]
    elif mode=="val":
        X, Y = X[int(len(X) * .8): -int(len(X) * .1)], Y[int(len(X) * .8): -int(len(X) * .1)]
    else:
        X, Y = X[-int(len(X) * .1):], Y[-int(len(X) * .1):]

    return X, Y

def get_batch_data():
    # Load data
    X, Y = load_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    x, y = tf.train.batch(input_queues,
                          num_threads=8,
                          batch_size=hp.batch_size,
                          capacity=hp.batch_size * 64,
                          allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()

