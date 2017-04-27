from collections import defaultdict, Counter
import numpy as np
import logging
import pickle as pkl
import os
import csv
import nltk
import codecs
import jieba
jieba.dt.tmp_dir='/home/wdxu/.tmp/'
logger = logging.getLogger(__name__)

UNK_TOKEN = 'UNK'
EOS_TOKEN = 'EOS'

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    proc_casetypeclf('../case_type_clf/raw/case_type_clf.csv', 
            '../case_type_clf/proc/log')
