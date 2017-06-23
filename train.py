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
from data_load import get_batch_data, load_vocab, load_data
from modules import *
from tqdm import tqdm

class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Load data
            self.x, self.y, self.num_batch = get_batch_data()  # (N, T)

            # Load vocabulary
            char2idx, idx2char = load_vocab()

            # Encoder
            ## Embedding
            enc = embedding(self.x,
                             vocab_size=len(char2idx),
                             num_units=hp.hidden_units,
                             scale=False,
                             scope="enc_embed")

            # Encoder pre-net
            prenet_out = prenet(enc,
                                num_units=[hp.hidden_units, hp.hidden_units//2],
                                dropout_rate=hp.dropout_rate,
                                is_training=is_training)  # (N, T, E/2)

            # Encoder CBHG
            ## Conv1D bank
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.hidden_units//2,
                               norm_type="ins",
                               is_training=is_training)  # (N, T, K * E / 2)

            ### Max pooling
            enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

            ### Conv1D projections
            enc = conv1d(enc, hp.hidden_units//2, 3, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type="ins", is_training=is_training, activation_fn=tf.nn.relu)
            enc = conv1d(enc, hp.hidden_units//2, 3, scope="conv1d_2")  # (N, T, E/2)
            enc += prenet_out  # (N, T, E/2) # residual connections

            ### Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc, num_units=hp.hidden_units//2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            ### Bidirectional GRU
            enc = gru(enc, hp.hidden_units//2, True)  # (N, T, E)

            # Final linear projection
            self.logits = tf.layers.dense(enc, 2) # 0 for non-space, 1 for space

            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.x, 0)) # masking
            self.num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget)
            self.num_targets = tf.reduce_sum(self.istarget)
            self.acc = self.num_hits / self.num_targets

            if is_training:
                # Loss
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # # Summary
                # tf.summary.scalar('mean_loss', self.mean_loss)
                # tf.summary.merge_all()



if __name__ == '__main__':
    # Construct graph
    g = Graph()
    print("Graph loaded")

    char2idx, idx2char = load_vocab()
    with g.graph.as_default():
        # For validation
        X_val, Y_val = load_data(mode="val")
        num_batch = len(X_val) // hp.batch_size

        # Start session
        sv = tf.train.Supervisor(graph=g.graph,
                                 logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                    # logging
                    if step % 100 == 0:
                        gs, mean_loss = sess.run([g.global_step, g.mean_loss])
                        print("\nAfter global steps %d, the training loss is %.2f" % (gs, mean_loss))

                # Save
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

                # Validation check
                total_hits, total_targets = 0, 0
                for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    x = X_val[step*hp.batch_size:(step+1)*hp.batch_size]
                    y = Y_val[step*hp.batch_size:(step+1)*hp.batch_size]
                    num_hits, num_targets = sess.run([g.num_hits, g.num_targets], {g.x: x, g.y: y})
                    total_hits += num_hits
                    total_targets += num_targets
                print("\nAfter epoch %d, the validation accuracy is %d/%d=%.2f" % (epoch, total_hits, total_targets, total_hits/total_targets))

    print("Done")


