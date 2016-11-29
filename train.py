# -*- coding: utf-8 -*-
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

from prepro import Hyperparams, load_data, load_embed_lookup_table
import sugartensor as tf

def get_batch_data(mode='train'):
    '''Makes batch queues from the data.
    
    Args:
      mode: A string. Either 'train', 'val', or 'test' 
    Returns:
      A Tuple of X_batch (Tensor), Y_batch (Tensor), and number of batches (int).
      X_batch and Y_batch have of the shape [batch_size, maxlen].
    '''
    # Load data
    X, Y = load_data(mode)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32), 
                                                  tf.convert_to_tensor(Y, tf.int32)])
    
    # create batch queues
    X_batch, Y_batch = tf.train.shuffle_batch( 
                                      input_queues,
                                      num_threads=8,
                                      batch_size=Hyperparams.batch_size, 
                                      capacity=Hyperparams.batch_size*64,
                                      min_after_dequeue=Hyperparams.batch_size*32, 
                                      allow_smaller_final_batch=False) 
    # calc total batch count
    num_batch = len(X) // Hyperparams.batch_size
    
    return X_batch, Y_batch, num_batch

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, mode="train"):
        '''
        Args:
          mode: A string. Either "train" , "val", or "test"
        '''
        if mode=='train':
            self.X_batch, self.Y_batch, self.num_batch = get_batch_data('train')
        else:
            self.X_batch = tf.placeholder(tf.int32, [Hyperparams.batch_size, Hyperparams.maxlen])
            self.Y_batch = tf.placeholder(tf.int32, [Hyperparams.batch_size, Hyperparams.maxlen])
        self.X_batch_rev = self.X_batch.sg_reverse_seq() # (8, 100)
          
        # make embedding matrix for input characters
        embed_mat = tf.convert_to_tensor(load_embed_lookup_table())
          
        # embed table lookup
        X_batch_3d = self.X_batch.sg_lookup(emb=embed_mat).sg_float() # (8, 100, 200)
        X_batch_rev_3d = self.X_batch_rev.sg_lookup(emb=embed_mat).sg_float() # (8, 100, 200)
        
        # 1st biGRU layer
        gru_fw1 = X_batch_3d.sg_gru(dim=Hyperparams.hidden_dim, ln=True) # (8, 100, 200)
        gru_bw1 = X_batch_rev_3d.sg_gru(dim=Hyperparams.hidden_dim, ln=True) # (8, 100, 200)
        gru1 = gru_fw1.sg_concat(target=gru_bw1) # (8, 100, 400)
        
        # 2nd biGRU layer
        gru_fw2 = gru1.sg_gru(dim=Hyperparams.hidden_dim*2, ln=True) # (8, 100, 400)
        gru_bw2 = gru1.sg_gru(dim=Hyperparams.hidden_dim*2, ln=True) # (8, 100, 400)
        gru2 = gru_fw2.sg_concat(target=gru_bw2) # (16, 100, 800)
        
        # fc dense layer
        reshaped = gru2.sg_reshape(shape=[-1, gru2.get_shape().as_list()[-1]])
        logits = reshaped.sg_dense(dim=3) # 1 for space 2 for non-space
        self.logits = logits.sg_reshape(shape=gru2.get_shape().as_list()[:-1] + [-1])
        
        if mode=='train':
            # cross entropy loss with logits ( for training set )
            self.loss = self.logits.sg_ce(target=self.Y_batch, mask=True)

            # accuracy evaluation ( for validation set )
            self.X_val_batch, self.Y_val_batch, self.num_batch = get_batch_data('val')
            
            self.acc = (self.logits.sg_reuse(input=self.X_val_batch)
                   .sg_accuracy(target=self.Y_val_batch, name='val'))
         
def train():
    g = ModelGraph()
    print "Graph loaded!"
    tf.sg_train(log_interval=10, loss=g.loss, eval_metric=[g.acc], max_ep=5, 
                save_dir='asset/train', early_stop=False, max_keep=10)
     
if __name__ == '__main__':
    train(); print "Done"