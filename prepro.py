#/usr/bin/python2
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

import numpy as np
import cPickle as pickle

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 8
    embed_dim = 200
    maxlen = 100
    hidden_dim = 200
    
def prepro():
    '''Embeds and vectorize words in corpus'''

    print "Create word2vec model"
    from nltk.corpus import brown
    sents = [" ".join(words) for words in brown.sents()]
    
    import os, gensim
    wv = gensim.models.Word2Vec(sents, min_count=1, size=Hyperparams.embed_dim)
    
    if not os.path.exists('data'): os.mkdir('data')
    wv.save('data/wv.m')
    
    print "Read and vectorize data ..." 
    vocab = ['<EMP>'] # <EMP> for zero padding
    xs, ys = [], []
    for sent in sents:
        if len(sent.replace(" ", "")) <= Hyperparams.maxlen:
            x, y = [], []
            for word in sent.split():
                for char in word:
                    if char not in vocab: vocab.append(char)
                    x.append(vocab.index(char))
                    y.append(2)
                y[-1] = 1 # 1 (space) for end of words
            y[-1] = 2 # 0 (no space) for end of sentence
            nb_pad_slots = Hyperparams.maxlen - len(x)
            x.extend(nb_pad_slots * [0]) # zero post-padding
            y.extend(nb_pad_slots * [0]) # zero post-padding
            
            xs.append(x)
            ys.append(y)

    # Convert to ndarrays    
    X = np.array(xs)
    Y = np.array(ys)

    # Split X, Y into train / val/ test. (8:1:1)
    X_train, X_val, X_test = X[: int(len(X)*.8)], X[int(len(X)*.8): -int(len(X)*.1)], X[-int(len(X)*.1):]
    Y_train, Y_val, Y_test = Y[: int(len(Y)*.8)], Y[int(len(Y)*.8): -int(len(Y)*.1)], Y[-int(len(Y)*.1):]       
    
    np.savez('data/train', X_train=X_train, Y_train=Y_train)
    np.savez('data/val', X_val=X_val, Y_val=Y_val)
    np.savez('data/test', X_test=X_test, Y_test=Y_test)
            
    # Make char maps from vocab
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    
    pickle.dump(idx2char, open('data/idx2char.pkl', 'wb'))
    pickle.dump(char2idx, open('data/char2idx.pkl', 'wb'))

def load_charmaps():
    '''Loads character dictionaries'''
    idx2char = pickle.load(open('data/idx2char.pkl', 'rb'))
    char2idx = pickle.load(open('data/char2idx.pkl', 'rb'))
    
    return idx2char, char2idx

def load_data(mode='train'):
    '''Loads numpy arrays
    Args:
      mode: A string. Either `train`, `val`, or `test`.
    '''
    data = np.load('data/{}.npz'.format(mode))
    return data['X_{}'.format(mode)], data['Y_{}'.format(mode)]

def load_word2vec_models():
    '''Loads word2vec model'''
    import gensim
    wv = gensim.models.Word2Vec.load('data/wv.m')
    
    return wv

def make_embed_lookup_table():
    '''Makes an embedding lookup table'''
    idx2char = load_charmaps()[0]
    wv = load_word2vec_models()
    
    def make_table(map, embed_dim, wv):
        table = np.random.randn(len(map), embed_dim)
        table[0, :] = 0 # for padding
        for idx in range(len(map)):
            char = map[idx]
            if char in wv:
                table[idx] = wv[char]
        return table
    
    lookup_table = make_table(idx2char, Hyperparams.embed_dim, wv)
    
    np.save('data/lookup_table', lookup_table)

def load_embed_lookup_table():
    '''Loads the embedding lookup table'''
    lookup_table = np.load('data/lookup_table.npy')
    
    return lookup_table

if __name__ == '__main__':
    prepro()
    make_embed_lookup_table()
    print "Done"        
