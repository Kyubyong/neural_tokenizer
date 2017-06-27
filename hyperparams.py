# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neural_tokenizer
'''

class Hyperparams:
    '''Hyperparameters'''

    # model
    maxlen = 150  # Maximum number of characters in a sentence. alias = T.
    minlen = 10 # Minimum number of characters in a sentence. alias = T.
    hidden_units = 256  # alias = E
    num_blocks = 6  # number of encoder/decoder blocks
    num_heads = 8
    dropout_rate = 0.2
    encoder_num_banks = 16
    num_highwaynet_blocks = 4


    # training
    num_epochs = 20
    batch_size = 128  # alias = N
    lr = 0.0001  # learning rate.
    logdir = 'logdir'  # log directory
    savedir = "results" # save directory





