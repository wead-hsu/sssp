# coding=utf-8

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
import re
import sys
import thulac

thuseg = thulac.thulac(seg_only=True)

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

def proc_gongshang_semi():
    unlabeled_data_fns = ['../gongshang/raw/unlabel/unlabel_data20-50.csv',
            '../gongshang/raw/unlabel/unlabel_data50-100.csv',
            '../gongshang/raw/unlabel/unlabel_data100-150.csv']

    data = pandas.read_csv('../gongshang/raw/feature20170515.csv', encoding='utf-8')
    data = data.iloc[np.random.permutation(len(data))]
    split_portion = [0.83, 0.08, 0.09]
    thresholds = [int(sum(split_portion[0:i]) * data.shape[0]) for i in range(4) ]
    train_df = data[thresholds[0]: thresholds[1]]
    valid_df = data[thresholds[1]: thresholds[2]]
    test_df = data[thresholds[2]: thresholds[3]]

    words = [w for l in list(train_df['Problem']) for w in jieba.cut(l)]
    for fn in unlabeled_data_fns:
        words.extend([w for l in open(fn) for w in jieba.cut(l)])
    word_cnt = dict(Counter(words))
    print('number of unique words: ', len(word_cnt))
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=15000)
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        """
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
        """
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
                #words = jieba.cut(clean_str(x.iloc[i]))
                words = jieba.cut((x.iloc[i]))
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                ys_i = ys.iloc[i].values.tolist()
                f.write('\t'.join([str(j+1) for j in ys_i]))
                f.write('\t' + ' '.join(ids))
                #f.write('\t' + x.iloc[i])
                #f.write('\t' + clean_str(x.iloc[i]))
                f.write('\n')

    save_data(train_df, '../gongshang/proc/labeled.data.idx', vocab, keys)
    save_data(valid_df, '../gongshang/proc/valid.data.idx', vocab, keys)
    save_data(test_df, '../gongshang/proc/test.data.idx', vocab, keys)

    with open('../gongshang/proc/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

    """ proc unlabeled data """
    portion = [1600, 1500, 800]
    portion = [p / sum(portion) for p in portion]
    num_unlabeled = 3900 * 2
    unlabeled_samples = []
    for idx, fn in enumerate(unlabeled_data_fns):
        with open(fn, 'r') as f:
            samples_tmp = f.read().strip().split('\n')
            samples_tmp = [samples_tmp[i] for i in np.random.permutation(len(samples_tmp))]
            unlabeled_samples.extend(samples_tmp[: int(portion[idx] * num_unlabeled)])
    unlabeled_samples = [unlabeled_samples[i] for i in np.random.permutation(len(unlabeled_samples))]
    tot = 0
    nun = 0
    with open('../gongshang/proc/unlabeled.data.idx', 'w') as f:
        UNK_ID = vocab[UNK_TOKEN]
        for s in unlabeled_samples:
            f.write('\t'.join(['0'] * 15))
            f.write('\t')
            words = [str(vocab.get(w, UNK_ID)) for w in jieba.cut(s)]
            if len(words) == 0:
                raise ('empty sentence found')
            tot += len(words)
            nun += sum([1 if w == str(UNK_ID) else 0 for w in words])
            f.write(' '.join(words))
            f.write('\n')
    print(tot, nun, nun/tot)

def proc_gongshang_clf():
    data = pandas.read_csv('../gongshang/raw/feature20170515.csv', encoding='utf-8')
    data = data.iloc[np.random.permutation(len(data))]
    split_portion = [0.83, 0.08, 0.09]
    thresholds = [int(sum(split_portion[0:i]) * data.shape[0]) for i in range(4) ]
    train_df = data[thresholds[0]: thresholds[1]]
    valid_df = data[thresholds[1]: thresholds[2]]
    test_df = data[thresholds[2]: thresholds[3]]

    words = [w for l in list(train_df['Problem']) for w in jieba.cut(l)]
    word_cnt = dict(Counter(words))
    print('number of unique words: ', len(word_cnt))
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=15000)
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        """
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
        """
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
                #words = jieba.cut(clean_str(x.iloc[i]))
                words = jieba.cut((x.iloc[i]))
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                ys_i = ys.iloc[i].values.tolist()
                f.write('\t'.join([str(j+1) for j in ys_i]))
                f.write('\t' + ' '.join(ids))
                #f.write('\t' + x.iloc[i])
                #f.write('\t' + clean_str(x.iloc[i]))
                f.write('\n')

    save_data(train_df, '../gongshang/clf/labeled.data.idx', vocab, keys)
    save_data(valid_df, '../gongshang/clf/valid.data.idx', vocab, keys)
    save_data(test_df, '../gongshang/clf/test.data.idx', vocab, keys)

    with open('../gongshang/clf/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

def proc_gongshang_semi3k():
    unlabeled_data_fns = ['../gongshang/raw/unlabel/unlabel_data20-50.csv',
            '../gongshang/raw/unlabel/unlabel_data50-100.csv',
            '../gongshang/raw/unlabel/unlabel_data100-150.csv']

    data = pandas.read_csv('../gongshang/raw/feature20170523.csv', encoding='utf-8')
    data = data.iloc[np.random.permutation(len(data))]
    split_portion = [0.83, 0.08, 0.09]
    thresholds = [int(sum(split_portion[0:i]) * data.shape[0]) for i in range(4) ]
    train_df = data[thresholds[0]: thresholds[1]]
    valid_df = data[thresholds[1]: thresholds[2]]
    test_df = data[thresholds[2]: thresholds[3]]

    words = [w for l in list(train_df['Problem']) for w in jieba.cut(l)]
    for fn in unlabeled_data_fns:
        words.extend([w for l in open(fn) for w in jieba.cut(l)])
    word_cnt = dict(Counter(words))
    print('number of unique words: ', len(word_cnt))
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=15000)
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        """
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
        """
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
                #words = jieba.cut(clean_str(x.iloc[i]))
                words = jieba.cut((x.iloc[i]))
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                ys_i = ys.iloc[i].values.tolist()
                f.write('\t'.join([str(j+1) for j in ys_i]))
                f.write('\t' + ' '.join(ids))
                #f.write('\t' + x.iloc[i])
                #f.write('\t' + clean_str(x.iloc[i]))
                f.write('\n')

    save_data(train_df, '../gongshang/semi3k/labeled.data.idx', vocab, keys)
    save_data(valid_df, '../gongshang/semi3k/valid.data.idx', vocab, keys)
    save_data(test_df, '../gongshang/semi3k/test.data.idx', vocab, keys)

    with open('../gongshang/semi3k/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

    """ proc unlabeled data """
    portion = [1600, 1500, 800]
    portion = [p / sum(portion) for p in portion]
    num_unlabeled = 3900 * 2
    unlabeled_samples = []
    for idx, fn in enumerate(unlabeled_data_fns):
        with open(fn, 'r') as f:
            samples_tmp = f.read().strip().split('\n')
            samples_tmp = [samples_tmp[i] for i in np.random.permutation(len(samples_tmp))]
            unlabeled_samples.extend(samples_tmp[: int(portion[idx] * num_unlabeled)])
    unlabeled_samples = [unlabeled_samples[i] for i in np.random.permutation(len(unlabeled_samples))]
    tot = 0
    nun = 0
    with open('../gongshang/semi3k/unlabeled.data.idx', 'w') as f:
        UNK_ID = vocab[UNK_TOKEN]
        for s in unlabeled_samples:
            f.write('\t'.join(['0'] * 15))
            f.write('\t')
            words = [str(vocab.get(w, UNK_ID)) for w in jieba.cut(s)]
            if len(words) == 0:
                raise ('empty sentence found')
            tot += len(words)
            nun += sum([1 if w == str(UNK_ID) else 0 for w in words])
            f.write(' '.join(words))
            f.write('\n')
    print(tot, nun, nun/tot)

def proc_gongshang_clf3k():
    data = pandas.read_csv('../gongshang/raw/feature20170523.csv', encoding='utf-8')
    data = data.iloc[np.random.permutation(len(data))]
    split_portion = [0.83, 0.08, 0.09]
    thresholds = [int(sum(split_portion[0:i]) * data.shape[0]) for i in range(4) ]
    train_df = data[thresholds[0]: thresholds[1]]
    valid_df = data[thresholds[1]: thresholds[2]]
    test_df = data[thresholds[2]: thresholds[3]]

    words = [w for l in list(train_df['Problem']) for w in jieba.cut(l)]
    word_cnt = dict(Counter(words))
    print('number of unique words: ', len(word_cnt))
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=15000)
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        """
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
        """
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
                #words = jieba.cut(clean_str(x.iloc[i]))
                words = jieba.cut((x.iloc[i]))
                ids = [str(vocab.get(w, UNK_ID)) for w in words]
                ys_i = ys.iloc[i].values.tolist()
                f.write('\t'.join([str(j+1) for j in ys_i]))
                f.write('\t' + ' '.join(ids))
                #f.write('\t' + x.iloc[i])
                #f.write('\t' + clean_str(x.iloc[i]))
                f.write('\n')

    save_data(train_df, '../gongshang/clf3k/labeled.data.idx', vocab, keys)
    save_data(valid_df, '../gongshang/clf3k/valid.data.idx', vocab, keys)
    save_data(test_df, '../gongshang/clf3k/test.data.idx', vocab, keys)

    with open('../gongshang/clf3k/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

def proc_zhongaonan_by_criteria():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070601/070601.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, label_map[l]])
                    cnt += 1
            if cnt == 0:
                shotlines.append([txt_i, 0])
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in jieba.cut(s[0])]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = jieba.cut(s[0])
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx+1] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案手段'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/tasks_hasnan', fn))
 
        with open(os.path.join('../zhongao/tasks_hasnan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/tasks_hasnan', fn))
 
        with open(os.path.join('../zhongao/tasks_hasnan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

def proc_zhongao_by_criteria():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070601/070601.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, label_map[l]])
                    cnt += 1
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in jieba.cut(s[0])]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = jieba.cut(s[0])
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values())))
        samples = proc_zhongao_by_criteria_help(['作案手段'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/tasks_nonan', fn))
  
        with open(os.path.join('../zhongao/tasks_nonan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values())))
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/tasks_nonan', fn))
 
        with open(os.path.join('../zhongao/tasks_nonan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

def proc_zhongaonan_by_criteria_for_hierachy():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070601/070601.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, 1, label_map[l]])
                    cnt += 1
            if cnt == 0:
                shotlines.append([txt_i, 0, 0])
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in jieba.cut(s[0])]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
    
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = jieba.cut(s[0])
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]) + '\t' + str(s[2]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    save_dir = '../zhongao/tasks_hasnan_hierachy'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案手段'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

def proc_0616_zhongaonan_by_criteria():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070616/070616.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['作案过程分析']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str or len(txt_i) < 5:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, label_map[l]])
                    cnt += 1
            if cnt == 0:
                shotlines.append([txt_i, 0])
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in list(zip(*thuseg.cut(s[0])))[0]]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = list(zip(*thuseg.cut(s[0])))[0]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx+1] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案手段1', '作案手段2', '作案手段3'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/proc/070616/tasks_hasnan', fn))
 
        with open(os.path.join('../zhongao/proc/070616/tasks_hasnan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2', '作案特点3'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/proc/070616/tasks_hasnan', fn))
 
        with open(os.path.join('../zhongao/proc/070616/tasks_hasnan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

def proc_0616_zhongao_by_criteria():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070616/070616.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['作案过程分析']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str or len(txt_i) < 5:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, label_map[l]])
                    cnt += 1
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in list(zip(*thuseg.cut(s[0])))[0]]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = list(zip(*thuseg.cut(s[0])))[0]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values())))
        samples = proc_zhongao_by_criteria_help(['作案手段1', '作案手段2', '作案手段3'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/proc/070616/tasks_nonan', fn))
  
        with open(os.path.join('../zhongao/proc/070616/tasks_nonan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values())))
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2', '作案特点3'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join('../zhongao/proc/070616/tasks_nonan', fn))
 
        with open(os.path.join('../zhongao/proc/070616/tasks_nonan', fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

def proc_0616_zhongaonan_by_criteria_for_hierachy():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070616/070616.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['作案过程分析']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str or len(txt_i) < 5:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, 1, label_map[l]])
                    cnt += 1
            if cnt == 0:
                shotlines.append([txt_i, 0, 0])
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in list(zip(*thuseg.cut(s[0])))[0]]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = list(zip(*thuseg.cut(s[0])))[0]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]) + '\t' + str(s[2]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    save_dir = '../zhongao/proc/070616/tasks_hasnan_hierachy'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案手段1', '作案手段2', '作案手段3'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2', '作案特点3'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

def proc_zhongaonan_inchar_by_criteria():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070601/070601.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, label_map[l]])
                    cnt += 1
            if cnt == 0:
                shotlines.append([txt_i, 0])
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in (s[0])]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = [w for w in (s[0])]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx+1] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    save_dir = '../zhongao/proc/070606/tasks_hasnan_inchar'
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案手段'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

def proc_zhongao_inchar_by_criteria():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070601/070601.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, label_map[l]])
                    cnt += 1
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in (s[0])]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = [w for w in s[0]]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    save_dir = '../zhongao/proc/070606/tasks_nonan_inchar'
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values())))
        samples = proc_zhongao_by_criteria_help(['作案手段'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
  
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values())))
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

def proc_zhongaonan_inchar_by_criteria_for_hierachy():
    def proc_zhongao_by_criteria_help(column_name, label_map):
        ifn = '../zhongao/raw/070601/070601.csv'
        df = pandas.read_csv(ifn)
        label = df[column_name]
        txt = df['案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            cnt = 0
            for l in label_i:
                if l in label_map:
                    shotlines.append([txt_i, 1, label_map[l]])
                    cnt += 1
            if cnt == 0:
                shotlines.append([txt_i, 0, 0])
            #if cnt == 2:
                #logger.warn('WARNING, has two labels: {} {}'.format(txt_i, label_i))
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        list_portion = [0.83, 0.06, 0.07]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in s[0]]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        print(vocab_size)
        
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = [w for w in s[0]]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(s[1]) + '\t' + str(s[2]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        os.mkdir(save_dir)
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)

    def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
            logger.debug(label_map)
        tgt_to_id = dict([v, idx] for idx, v in enumerate(sorted(set(label_map.values()))))
        logger.debug(tgt_to_id)
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        logger.debug(label_map)
        return label_map, tgt_to_id
    
    criteria_dir = '../zhongao/raw/split_criteria/approach'
    save_dir = '../zhongao/proc/070606/tasks_hasnan_inchar_hierachy/'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案手段'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

    criteria_dir = '../zhongao/raw/split_criteria/characteristic'
    for fn in os.listdir(criteria_dir):
        print(fn)
        label_map, tgt_to_id = get_label_map(os.path.join(criteria_dir, fn))
        print(len(set(label_map.values()))+1)
        samples = proc_zhongao_by_criteria_help(['作案特点1','作案特点2'], label_map)
        logger.info('task: {}, number of samples: {}'.format(fn, len(samples)))
        data_to_idx(samples, os.path.join(save_dir, fn))
 
        with open(os.path.join(save_dir, fn) + '/label_map', 'w') as f:
            for l in label_map:
                f.write(l + '\t' + str(label_map[l]) + '\n')

        with open(os.path.join(save_dir, fn) + '/labels.pkl', 'wb') as f:
            pkl.dump(tgt_to_id, f)

def proc_semeval2010task8():
    rawdata_train_fn = '../semeval2018task8/raw/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    rawdata_test_fn = '../semeval2018task8/raw/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    def read_rawdata(fn):
        samples = []
        with open(fn, 'r') as f:
            lines = f.read().split('\n')
            for i in range(len(lines)//4):
                txt = lines[i*4].strip().split('\t')[1][1:-1]
                txt = txt.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','')
                label = lines[i*4+1].strip().split('(')[0]
                samples.append([txt,label])
        return samples

    train_data = read_rawdata(rawdata_train_fn)
    train_data = [train_data[i] for i in np.random.permutation(len(train_data))]
    train_data, valid_data = train_data[400:], train_data[:400]
    test_data = read_rawdata(rawdata_test_fn)

    words = [w for sample in train_data for w in wt(sample[0])]
    word_cnt = Counter(words)
    vocab, vocab_size = build_vocab(word_cnt, max_vocab_size=15000)
    labels = set([sample[1] for sample in train_data])
    labels.remove('Other')
    label2idx = dict([[v, i+1] for i, v in enumerate(labels)])
    label2idx['Other'] = 0

    def save_data(samples, vocab, ofn):
        UNK_ID = vocab[UNK_TOKEN]
        with open(ofn, 'w') as f:
            for s in samples:
                words = [w for w in wt(s[0])]
                word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                f.write(str(label2idx[s[1]]))
                f.write('\t' + ' '.join(word_idx))
                f.write('\n')
    
    save_dir = '../semeval2018task8/proc/'
    #os.mkdir(save_dir)
    splits = [train_data, valid_data, test_data]
    names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
    for i in range(3):
        data = splits[i]
        name = names[i]
        save_data(data, vocab, os.path.join(save_dir, name))

    with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
        pkl.dump(vocab, f)

    with open(os.path.join(save_dir, 'label2idx.pkl'), 'wb') as f:
        pkl.dump(label2idx, f)

def proc_zhongao_zzxs(inchar, cv_idx):
    def proc_zhongao_by_criteria_help():
        ifn = '../zhongao/raw/zzxs/070704.xlsx'
        df = pandas.read_excel(ifn)
        label = df['标签1']
        txt = df['脱敏案情']
        shotlines = []
        for i in range(label.shape[0]):
            label_i = label.iloc[i].tolist()
            txt_i = txt.iloc[i]
            if type(txt_i) != str:
                continue
            shotlines.append([txt_i, str(label_i)])
        return shotlines
    
    def data_to_idx(samples, save_dir):
        samples = [samples[i] for i in np.random.permutation(len(samples))]
        samples = samples[int(len(samples) * 0.1 * cv_idx):] + samples[:int(len(samples) * 0.1 * cv_idx)]
        list_portion = [0.80, 0.10, 0.10]
        divide_pos = [int(sum(list_portion[:i])*len(samples)) for i in range(len(list_portion)+1)]
        splits = [samples[divide_pos[i]: divide_pos[i+1]] for i in range(len(list_portion))]
        
        for s in splits[0]:
            if type(s[0]) != str:
                print(s)
        words = [w for s in splits[0] for w in (s[0])]
        if not inchar: words = [w for s in splits[0] for w in list(zip(*thuseg.cut(s[0])))[0]]
        vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=15000)
        classes = set([s[1] for s in samples])
        class_map = dict([[v, idx+1] for idx, v in enumerate(sorted(classes)) if v != 'nan'])
        class_map['nan'] = 0
        logger.info(class_map)
        logger.info(vocab_size)
        
        os.mkdir(save_dir)
        def save_data(samples, vocab, ofn):
            UNK_ID = vocab[UNK_TOKEN]
            with open(ofn, 'w') as f:
                for s in samples:
                    words = [w for w in s[0]]
                    if not inchar: words = list(zip(*thuseg.cut(s[0])))[0]
                    word_idx = [str(vocab.get(w, UNK_ID)) for w in words]
                    f.write(str(class_map[s[1]]))
                    f.write('\t' + ' '.join(word_idx))
                    f.write('\n')
    
        names = ['train.data.idx', 'valid.data.idx', 'test.data.idx']
        for i in range(3):
            data = splits[i]
            name = names[i]
            save_data(data, vocab, os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)
        with open(os.path.join(save_dir, 'class_map.pkl'), 'wb') as f:
            pkl.dump(class_map, f)


    np.random.seed(0)
    save_dir = '../zhongao/proc/zzxs/070704/inchar_cv{}'.format(cv_idx)
    if not inchar: save_dir = '../zhongao/proc/zzxs/070704/inword_cv{}'.format(cv_idx)
    samples = proc_zhongao_by_criteria_help()
    logger.info('task: {}, number of samples: {}'.format('zzxs', len(samples)))
    data_to_idx(samples, save_dir)
  
def proc_googleextration():
    rawfiles_dir = '../google-extraction/raw'
    save_dir = '../google-extraction/proc'
    rawfilenames = ['../google-extraction/raw/20130403-institution.json',
            '../google-extraction/raw/20130403-place_of_birth.json',
            '../google-extraction/raw/20131104-date_of_birth.json',
            '../google-extraction/raw/20131104-education-degree.json',
            '../google-extraction/raw/20131104-place_of_death.json']

    labels = ['institution',
            'place_of_birth',
            'date_of_birth',
            'education',
            'place_of_death',]

    samples = OrderedDict()
    for i in range(5):
        fn = rawfilenames[i]
        label = labels[i]
        with open(fn, 'r') as f:
            tmp_samples = [json.loads(l.replace('\\', '')) for l in f]
            tmp_x = [s['evidences'][0]['snippet'] for s in tmp_samples]
            samples[label] = tmp_x
            logger.info('Labelname: {} Number of samples: {}'.format(label, len(tmp_x)))
    
    samples = {k: samples[k][:10000] for k in samples}

    def split_dict_samples(samples, list_split_portion):
        num_splits = len(list_split_portion)
        splits = [{} for i in range(num_splits)]
        for k in samples:
            num_samples = len(samples[k])
            list_threshold = [int(sum(list_split_portion[:i])*num_samples) for i in range(4)]
            split_k = [samples[k][list_threshold[i]: list_threshold[i+1]] for i in range(3)]
            for i in range(num_splits):
                splits[i][k] = split_k[i]
        return splits
    
    # split samples into [train, valid, test] {{{
    train, valid, test = split_dict_samples(samples, list_split_portion=[0.90, 0.05, 0.05])
    for k in samples:
        logger.info('len(train[{}])={}'.format(k, len(train[k])))
        logger.info('len(valid[{}])={}'.format(k, len(valid[k])))
        logger.info('len(test[{}])={}'.format(k, len(test[k])))

    train_flt = [(x, y) for y in train for x in train[y]]
    valid_flt = [(x, y) for y in valid for x in valid[y]]
    test_flt = [(x, y) for y in test for x in test[y]]
    # }}}

    # build vocab {{{
    words = [w for sample in train_flt for w in wt(sample[0])]
    vocab, vocab_size = build_vocab(Counter(words), max_vocab_size=20000)
    label_dict = {labels[idx]: idx for idx in range(5)}
    # }}}

    def save_data(data_list, save_dir, vocab, label_dict):
        fns = ['train', 'valid', 'test', 'labeled', 'unlabeled']
        UNK_ID = vocab[UNK_TOKEN]
        #os.mkdir(save_dir)
        for i in range(5):
            with open(os.path.join(save_dir, fns[i] + '.data.idx'), 'w') as f:
                for sample in data_list[i]:
                    words_tmp = wt(sample[0])
                    label_tmp = sample[1]
                    widx = [str(vocab.get(w, UNK_ID)) for w in words_tmp]
                    lidx = str(label_dict[label_tmp])
                    f.write(lidx)
                    f.write('\t')
                    f.write(' '.join(widx))
                    f.write('\n')
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(vocab, f)
        with open(os.path.join(save_dir, 'label.pkl'), 'wb') as f:
            pkl.dump(label_dict, f)
    
    # create dataset with different number of labeled samples {{{
    for label_portion in [0.05, 0.1, 0.2, 0.4]:
        labeled, unlabeled = split_dict_samples(train, list_split_portion=[label_portion, 1-label_portion])
        labeled_flt = [(x, y) for y in labeled for x in labeled[y]]
        unlabeled_flt = [(x, y) for y in labeled for x in unlabeled[y]]
        logger.info('labeled_portion: {}'.format(label_portion))
        for k in samples:
            logger.info('len(labeled[{}])={}'.format(k, len(labeled[k])))
            logger.info('len(unlabeled[{}])={}'.format(k, len(unlabeled[k])))

        data_list = [train_flt, valid_flt, test_flt, labeled_flt, unlabeled_flt]
        save_data(data_list, os.path.join(save_dir, str(label_portion)), vocab, label_dict)
    # }}}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #proc_casetypeclf('../case_type_clf/raw/case_type_clf.csv', '../case_type_clf/proc/log')
    #proc_beer_for_reg()
    #proc_beer_for_clf()
    #proc_agnews()
    #proc_gongshang_semi()
    #proc_gongshang_clf()
    #proc_gongshang_semi3k()
    #proc_gongshang_clf3k()
    #proc_zhongao_by_criteria()
    #proc_zhongaonan_by_criteria()
    #proc_zhongaonan_by_criteria_for_hierachy()
    #proc_0616_zhongao_by_criteria()
    #proc_0616_zhongaonan_by_criteria()
    #proc_0616_zhongaonan_by_criteria_for_hierachy()
    #proc_zhongao_inchar_by_criteria()
    #proc_zhongaonan_inchar_by_criteria()
    #proc_zhongaonan_inchar_by_criteria_for_hierachy()
    #proc_semeval2010task8()
    for i in range(10): proc_zhongao_zzxs(inchar=True, cv_idx=i)
    for i in range(10): proc_zhongao_zzxs(inchar=False, cv_idx=i)
    #proc_googleextration()
