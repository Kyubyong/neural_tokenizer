# -*- coding: utf-8 -*-
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_data, load_charmaps
from train import ModelGraph
import codecs

def main():  
    graph = ModelGraph("test")
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
                     
        X, Y = load_data("test")
        idx2chr = load_charmaps()[0]
        
        with codecs.open('results.txt', 'w', 'utf-8') as fout:
            results = []
            for step in range(len(X) // Hyperparams.batch_size -1):
                X_batch = X[step: step + Hyperparams.batch_size, :] 
                Y_batch = Y[step: step + Hyperparams.batch_size, :]
                
                # predict characters
                logits = sess.run(graph.logits, {graph.X_batch: X_batch})
                preds = np.squeeze(np.argmax(logits, -1))
                
                for x, y, p in zip(X_batch, Y_batch, preds): # sentence-wise
                    ground_truth = ''
                    predicted = ''
                    for xx, yy, pp in zip(x, y, p): # character-wise
                        if xx == 0: break
                        else: 
                            predicted += idx2chr.get(xx, "*")
                            ground_truth += idx2chr.get(xx, "*")
                        if pp == 1: predicted += " "
                        if yy == 1: ground_truth += " "
                        
                        if pp == yy: results.append(1)
                        else: results.append(0)
                        
                    fout.write(u"▌Expected: " + ground_truth + "\n")
                    fout.write(u"▌Got: " + predicted + "\n\n")
            fout.write(u"Final Accuracy = %d/%d=%.2f" % ( sum(results), len(results), float(sum(results)) / len(results)) )
                                        
if __name__ == '__main__':
    main()
    print "Done"

