import pickle as pkl
import os
import numpy as np
from shutil import copyfile

def convert(filename, embd_fn, save_dir):
    data = pkl.load(open(filename, 'rb'), encoding='latin')
    print(data.keys())
    x_l, y_l, x_u, x_dev, y_dev, x_test, y_test, vocab, vocab_size = data['s_l_train'], data['y_l_train'],\
            data['s_u'], data['s_l_dev'], data['y_l_dev'], data['s_l_test'], data['y_l_test'], data['wdict'],\
            data['dict_size']
    print(type(x_l))
    y_u = [0]*len(x_u)

    def _save(x, y, f):
        for i in range(len(y)):
            f.write(str(np.argmax(y[i])))
            f.write('\t')
            for j in x[i]:
                for k in j:
                    f.write(str(k) + ' ')
            f.write('\n')
        return

    with open(os.path.join(save_dir, 'embd.pkl'), 'wb') as f:
        embd = pkl.load(open(embd_fn, 'rb'), encoding='latin')['vec_norm']
        pkl.dump(embd, f)

    with open(os.path.join(save_dir, 'labeled.data.idx'), 'w') as f:
        _save(x_l, y_l, f)

    with open(os.path.join(save_dir, 'unlabeled.data.idx'), 'w') as f:
        _save(x_u, y_u, f)

    with open(os.path.join(save_dir, 'valid.data.idx'), 'w') as f:
        _save(x_dev, y_dev, f)

    with open(os.path.join(save_dir, 'test.data.idx'), 'w') as f:
        _save(x_test, y_test, f)

    with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
        pkl.dump(vocab, f)

    with open(os.path.join(save_dir, 'vocab_size'), 'w') as f:
        f.write(str(vocab_size))

convert('imdb.semi.2500', 'imdb.20k.glove.300', '/home/wdxu/codes/sssp/data/imdb/imdb2500')
convert('imdb.semi.5000', 'imdb.20k.glove.300', '/home/wdxu/codes/sssp/data/imdb/imdb20000')
convert('imdb.semi.10000', 'imdb.20k.glove.300', '/home/wdxu/codes/sssp/data/imdb/imdb10000')
convert('imdb.semi.15000', 'imdb.20k.glove.300', '/home/wdxu/codes/sssp/data/imdb/imdb15000')
convert('imdb.semi.20000', 'imdb.20k.glove.300', '/home/wdxu/codes/sssp/data/imdb/imdb20000')

convert('ag.semi.8k', 'ag.23k.glove.300', '/home/wdxu/codes/sssp/data/agnews/ag8000')
convert('ag.semi.16k', 'ag.23k.glove.300', '/home/wdxu/codes/sssp/data/agnews/ag16000')
convert('ag.semi.32k', 'ag.23k.glove.300', '/home/wdxu/codes/sssp/data/agnews/ag32000')
