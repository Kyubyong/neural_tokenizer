# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neural_tokenizer
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from data_load import get_batch_data, load_vocab, load_data


def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Y = load_data(mode="test")  # texts
    char2idx, idx2char = load_vocab()

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # Inference
            if not os.path.exists(hp.savedir): os.mkdir(hp.savedir)
            with open("{}/{}".format(hp.savedir, mname), 'w') as fout:
                results = []
                baseline_results = []
                for step in range(len(X) // hp.batch_size):
                    x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
                    y = Y[step * hp.batch_size: (step + 1) * hp.batch_size]

                    # predict characters
                    preds = sess.run(g.preds, {g.x: x})

                    for xx, yy, pp in zip(x, y, preds):  # sentence-wise
                        expected = ''
                        got = ''
                        for xxx, yyy, ppp in zip(xx, yy, pp):  # character-wise
                            if xxx == 0:
                                break
                            else:
                                got += idx2char.get(xxx, "*")
                                expected += idx2char.get(xxx, "*")
                            if ppp == 1: got += " "
                            if yyy == 1: expected += " "

                            # prediction results
                            if ppp == yyy:
                                results.append(1)
                            else:
                                results.append(0)

                            # baseline results
                            if yyy == 0: # no space
                                baseline_results.append(1)
                            else:
                                baseline_results.append(0)

                        fout.write("▌Expected: " + expected + "\n")
                        fout.write("▌Got: " + got + "\n\n")
                fout.write(
                    "Final Accuracy = %d/%d=%.4f\n" % (sum(results), len(results), float(sum(results)) / len(results)))
                fout.write(
                    "Baseline Accuracy = %d/%d=%.4f" % (sum(baseline_results), len(baseline_results), float(sum(baseline_results)) / len(baseline_results)))



if __name__ == '__main__':
    eval()
    print("Done")

