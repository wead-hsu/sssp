from collections import defaultdict, Counter
import numpy as np
import logging
import pickle as pkl
import os
import csv
import nltk
import codecs
import jieba
from nltk.tokenize import word_tokenize as wt
import pandas

jieba.dt.tmp_dir='/home/wdxu/.tmp/'
logger = logging.getLogger(__name__)

UNK_TOKEN = 'UNK'
EOS_TOKEN = 'EOS'

np.random.seed(0)

def build_vocab(word_cnt, cnt_threshold=0, max_vocab_size=0):

    """ Build vocabulary from a single file.
    Assumption:
        The data is saved in a single file.
        Each line in the file represents a sample.

    Args:
        cnt_threshold: if wcnt[k] < cnt_threshold, k will not 
            be included.
        vocab_size: set max_vocab_size.
    Returns:
        vocab: map from word to idx.
        vocab_size: size of vocabulary
    """

    all_sorted_word_cnt = sorted(word_cnt.items(), key=lambda x: (-x[1], x[0]))
    sub_sorted_word_cnt = all_sorted_word_cnt
    if max_vocab_size: 
        sub_sorted_word_cnt = sub_sorted_word_cnt[:max_vocab_size-2]
    sub_sorted_word_cnt = [(k, v) for k, v in sub_sorted_word_cnt if v >= cnt_threshold]
    num_all_words = np.sum([v for k, v in all_sorted_word_cnt])
    num_sub_words = np.sum([v for k, v in sub_sorted_word_cnt])
    logger.info('Coverage Percentage: {0} {1} {2}'.format(num_all_words, num_sub_words, num_sub_words*1.0/ num_all_words))
    words, _ = list(zip(*sub_sorted_word_cnt))
    words = [EOS_TOKEN, UNK_TOKEN] + list(words)
    vocab = dict(zip(words, range(len(words))))
    vocab_size = len(words)
    logger.info('Size of vocabulary: {0}'.format(vocab_size))

    return vocab, vocab_size

def get_classes():
    pass

def proc_casetypeclf(ifn, logfn='log'):
    data = []
    with codecs.open(ifn, 'rU') as inf:
        inf = csv.reader(inf, delimiter=',')
        l = inf.next()
        for l in inf:
            data.append([token.decode('utf-8') for token in l])
    
    classes = [sample[1] for sample in data]
    class_cnt = Counter(classes)
    sorted_class_cnt = sorted(dict(class_cnt).items(), key=lambda x: x[1], reverse=True)
    classname_to_id = dict([[item[0], idx] for idx, item in enumerate(sorted_class_cnt)])
    #for name in classname_to_id:
        #print(name + '\t' + str(classname_to_id[name]))
    
    # log type distribution
    with codecs.open(logfn, 'w', 'utf-8') as f:
        for l in sorted_class_cnt:
            f.write(l[0] + '\t' + str(l[1]) + '\n')

    # cluster the samples with same label
    cluster = defaultdict(list)
    for sample in data:
        cluster[sample[1]].append(sample[0])
    
    # split train dev test
    portion = [0.8, 0.1, 0.1]
    train = {}
    dev = {}
    test = {}
    for key in cluster:
        num_samples_with_key = len(cluster[key])
        train_threshold = int(num_samples_with_key * sum(portion[:1]))
        dev_threshold = int(num_samples_with_key * sum(portion[:2]))
        test_threshold = int(num_samples_with_key * sum(portion[:3]))
        train[key] = cluster[key][:train_threshold]
        dev[key] = cluster[key][train_threshold: dev_threshold]
        test[key] = cluster[key][dev_threshold: test_threshold]

    # build dict
    text_samples_in_train = [sample for key in train for sample in train[key]]
    words = [w for sample in text_samples_in_train for w in jieba.cut(sample)]
    word_cnt = Counter(words)
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=20000)
    with open('../case_type_clf/proc/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f, protocol=2)
    #print(word_cnt.keys()[1])

    # split labeled and labeled data from train data
    portion = [0.2, 0.8]
    labeled_from_train = {}
    unlabeled_from_train = {}
    for key in train:
        num_samples_with_key = len(train[key])
        labeled_threshold = int(num_samples_with_key * sum(portion[:1]))
        unlabeled_threshold = int(num_samples_with_key * sum(portion[:2]))
        labeled_from_train[key] = train[key][:labeled_threshold]
        unlabeled_from_train[key] = train[key][labeled_threshold: unlabeled_threshold]

    # dump data
    def save_data(data, vocab, c2d, ofn):
        all_cnt = 0
        unk_cnt = 0
        data = [[key, text] for key in data for text in data[key]]
        data = [data[i] for i in np.random.permutation(len(data))]
        UNK_ID = vocab[UNK_TOKEN]
        lens = []
        with open(ofn, 'w') as f:
            for sample in data:
                classname = str(c2d[sample[0]])
                words = [str(vocab.get(w, UNK_ID)) for w in jieba.cut(sample[1])]
                if len(words) > 100:
                    continue
                lens.append(len(words))
                all_cnt += len(words)
                unk_cnt += len([1 for w in words if w != '1'])
                f.write(classname)
                f.write('\t')
                f.write(' '.join(words))
                f.write('\n')
        logger.info('Coverage: {}'.format( unk_cnt * 1.0 / all_cnt))
        logger.info(np.histogram(lens))
        logger.info(np.mean(lens))
        logger.info(np.std(lens))
        
    save_data(train, vocab, classname_to_id, '../case_type_clf/proc/train_all.data.idx')
    save_data(labeled_from_train, vocab, classname_to_id, '../case_type_clf/proc/labeled.data.idx')
    save_data(unlabeled_from_train, vocab, classname_to_id, '../case_type_clf/proc/unlabeled.data.idx')
    save_data(dev, vocab, classname_to_id, '../case_type_clf/proc/dev.data.idx')
    save_data(test, vocab, classname_to_id, '../case_type_clf/proc/test.data.idx')

def proc_beer_for_reg():
    ifn = '../beer/raw/reviews.aspect1.train.txt'
    word_cnt = defaultdict(int)
    with open(ifn, 'r') as f:
        words = [w  for l in f for w in l.split('\t')[1].split()]
        word_cnt = Counter(words)
        vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=20000)
    with open('../beer/proc/cls_0-aspect_1/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f, protocol=2)
    
    train_fn = '../beer/raw/reviews.aspect1.train.txt'
    test_fn = '../beer/raw/reviews.aspect1.heldout.txt'
    with open(train_fn, 'r') as f:
        samples = f.read().strip().split('\n')
    samples = [samples[idx] for idx in np.random.permutation(len(samples))]
    portion = [0.1, 0.8, 0.1]
    thresholds = [int(len(samples) * sum(portion[:idx+1])) for idx in range(len(portion))]
    labeled_data = samples[0: thresholds[0]]
    unlabeled_data = samples[thresholds[0]: thresholds[1]]
    dev_data = samples[thresholds[1]: thresholds[2]]

    with open(test_fn, 'r') as f:
        test_data = f.read().strip().split('\n')

    def save_data(data, ofn, vocab, cid):
        UNK_ID = vocab.get(UNK_TOKEN)
        with open(ofn, 'w') as ofh:
            for l in data:
                values = [n for n in l.split('\t')[0].split()]
                words = l.split('\t')[1].split()
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                ofh.write(values[cid] + '\t' + ' '.join(ids) + '\n')

    save_data(samples, '../beer/proc/cls_0-aspect_1/train_all.data.idx', vocab, 0)
    save_data(labeled_data, '../beer/proc/cls_0-aspect_1/labeled.data.idx', vocab, 0)
    save_data(unlabeled_data, '../beer/proc/cls_0-aspect_1/unlabeled.data.idx', vocab, 0)
    save_data(dev_data, '../beer/proc/cls_0-aspect_1/dev.data.idx', vocab, 0)
    save_data(test_data, '../beer/proc/cls_0-aspect_1/test.data.idx', vocab, 0)

def proc_beer_for_clf():
    ifn = '../beer/raw/reviews.aspect1.train.txt'
    word_cnt = defaultdict(int)
    with open(ifn, 'r') as f:
        words = [w  for l in f for w in l.split('\t')[1].split()]
        word_cnt = Counter(words)
        vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=20000)
    with open('../beer/proc/cls_0-aspect_1/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f, protocol=2)
    
    train_fn = '../beer/raw/reviews.aspect1.train.txt'
    test_fn = '../beer/raw/reviews.aspect1.heldout.txt'
    with open(train_fn, 'r') as f:
        samples = f.read().strip().split('\n')
    samples = [samples[idx] for idx in np.random.permutation(len(samples))]
    portion = [0.2, 0.7, 0.1]
    thresholds = [int(len(samples) * sum(portion[:idx+1])) for idx in range(len(portion))]
    labeled_data = samples[0: thresholds[0]]
    unlabeled_data = samples[thresholds[0]: thresholds[1]]
    dev_data = samples[thresholds[1]: thresholds[2]]

    with open(test_fn, 'r') as f:
        test_data = f.read().strip().split('\n')

    def save_data(data, ofn, vocab, cid):
        UNK_ID = vocab.get(UNK_TOKEN)
        with open(ofn, 'w') as ofh:
            for l in data:
                values = [float(n) for n in l.split('\t')[0].split()]
                words = l.split('\t')[1].split()
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                if values[cid] <= 0.7:
                    ofh.write('0\t' + ' '.join(ids) + '\n')
                elif values[cid] >= 0.8:
                    ofh.write('1\t' + ' '.join(ids) + '\n')
                else:
                    pass

    save_path = '../beer/proc/cls_0-aspect_1-clf-0.2/'
    save_data(samples, save_path + 'train_all.data.idx', vocab, 0)
    save_data(labeled_data, save_path + 'labeled.data.idx', vocab, 0)
    save_data(unlabeled_data, save_path + 'unlabeled.data.idx', vocab, 0)
    save_data(dev_data, save_path + 'dev.data.idx', vocab, 0)
    save_data(test_data, save_path + 'test.data.idx', vocab, 0)

def proc_agnews():
    train_fn = '../ag_news/raw/train.csv'
    test_fn = '../ag_news/raw/test.csv'

    def read_csv(filename):
        x, y = [], []
        with open(filename, 'r') as f:
            cr = csv.reader(f, delimiter=',')
            for l in cr:
                x.append(l[2].replace('\\', ' ').lower())
                y.append(int(l[0])-1)   # [1, n] in raw file
        return (x, y)

    def split_data(xy, num_classes, portion=0.2):
        x_c = []
        y_c = []
        # 1: n in raw datafile
        for c in range(num_classes):
            indices = np.where(np.asarray(xy[1]) == c)[0]
            x_c.append([xy[0][idx] for idx in indices])
            y_c.append([xy[1][idx] for idx in indices])
        
        train_x, train_y, valid_x, valid_y = [], [], [], []

        num_train_sample = int(len(x_c[0]) * (1 - portion))
        for idx in range(num_classes):
            x_ci = x_c[idx]
            y_ci = y_c[idx]
            train_x += x_ci[:num_train_sample]
            train_y += y_ci[:num_train_sample]
            valid_x += x_ci[num_train_sample:]
            valid_y += y_ci[num_train_sample:]

        return (train_x, train_y), (valid_x, valid_y)


    train_data = read_csv(train_fn)
    test_data = read_csv(test_fn)

    word_cnt = dict(Counter([w for s in train_data[0] for w in wt(s)]))
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=20000)
    train_data, valid_data = split_data(train_data, 4, 0.02)
    labeled_data, unlabeled_data = split_data(train_data, 4, 0.9)

    def save_data(data, ofn):
        UNK_ID = vocab[UNK_TOKEN]
        with open(ofn, 'w') as f:
            for i in range(len(data[0])):
                widx = [str(vocab.get(w, UNK_ID)) for w in wt(data[0][i])]
                f.write(str(data[1][i]) + '\t')
                f.write(' '.join(widx) + '\n')

    save_data(train_data, '../ag_news/proc/train_all.data.idx')
    save_data(labeled_data, '../ag_news/proc/labeled.data.idx')
    save_data(unlabeled_data, '../ag_news/proc/unlabeled.data.idx')
    save_data(valid_data, '../ag_news/proc/valid.data.idx')
    save_data(test_data, '../ag_news/proc/test.data.idx')

    with open('../ag_news/proc/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

def proc_gongshang():
    data = pandas.read_csv('../gongshang/feature20170515.csv', encoding='utf-8')
    split_portion = [0.83, 0.08, 0.09]
    thresholds = [int(sum(split_portion[0:i]) * data.shape[0]) for i in range(4) ]
    train_df = data[thresholds[0]: thresholds[1]]
    valid_df = data[thresholds[1]: thresholds[2]]
    test_df = data[thresholds[2]: thresholds[3]]

    words = [w for l in list(train_df['Problem']) for w in jieba.cut(l)]
    word_cnt = dict(Counter(words))
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=10000)
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    
    keys = ['GetPay', 'AssoPay', 'WorkTime', 'WorkPlace', 'JobRel', 'DiseRel', 'OutForPub',
            'OnOff', 'InjIden', 'EndLabor', 'LaborContr', 'ConfrmLevel', 'Level', 'Insurance', 'HaveMedicalFee']

    def save_data(df, ofn, vocab, keys):
        UNK_ID = vocab[UNK_TOKEN]
        with open(ofn, 'w') as f:
            x = df['Problem']
            ys = df[keys]
            for i in range(x.count()):
                #print(x.iloc[i])
                words = jieba.cut(x.iloc[i])
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                ys_i = ys.iloc[i].values.tolist()
                f.write('\t'.join([str(j+1) for j in ys_i]))
                f.write('\t' + ' '.join(ids) + '\n')

    save_data(train_df, '../gongshang/proc/labeled.data.idx', vocab, keys)
    save_data(valid_df, '../gongshang/proc/valid.data.idx', vocab, keys)
    save_data(test_df, '../gongshang/proc/test.data.idx', vocab, keys)

    with open('../gongshang/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #proc_casetypeclf('../case_type_clf/raw/case_type_clf.csv', '../case_type_clf/proc/log')
    #proc_beer_for_reg()
    #proc_beer_for_clf()
    #proc_agnews()
    proc_gongshang()
