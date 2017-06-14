import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pickle as pkl
from pylab import *  
from itertools import cycle
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['axes.unicode_minus'] = False
import pandas
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')

from sssp.run.load_clf import *
pred, label, args = main()
"""

with open('res.pkl', 'rb') as f:
    pred = pkl.load(f)
    label = pkl.load(f)
    args = pkl.load(f)

"""
pred = pred.squeeze()

from sklearn.metrics import roc_curve, auc

def get_label_map(fn):
        with open(fn, 'r') as f:
            label_map = {}
            tgt = ''
            for l in f:
                key, value = l.split('\t')
                if value != '\n': tgt = value
                label_map[key] = tgt
        tgt_to_id = dict([v, idx+1] for idx, v in enumerate(sorted(set(label_map.values()))))
        label_map = dict([key, tgt_to_id[label_map[key]]] for key in label_map)
        return label_map

def get_label_map0(path):
    label_map = {}
    with open(path, 'r') as f:
        prev = ''
        for line in f:
            key, value = line.split('\t')
            if value != '\n': prev = value
            label_map[key] = prev[:-1]
    return label_map

def plot_roc_single_label(pred, label):
    #print(pred, label)
    #for threshold in np.arange(0, 1, 0.05):
    a, b, _ = roc_curve(label, pred)
    roc_auc = auc(a, b)
    #print(a,b)
    print(roc_auc)

task_name = [w for w in args.test_path.split('/') if w.startswith('Code')][0]
label_map_path = ['data/zhongao/raw/split_criteria/approach/' + task_name, 
                'data/zhongao/raw/split_criteria/characteristic/' + task_name]
label_map_path = [fn for fn in label_map_path if os.path.exists(fn)][0]
label_map = get_label_map(label_map_path)
label_map['NAN'] = 0
labelname = dict([[v, k] for k, v in label_map.items()])
print(label_map)
print(labelname)
print(task_name)

def plot():
    #plt.plot(label)
    #plt.savefig('tmp.png')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(label==i, pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
        #plot_roc_single_label(pred[:, i], label==i)

    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(args.num_classes), colors):
        print('ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot(fpr[i], tpr[i], color=np.random.rand(3,1), lw=lw,
                label='{0} (area = {1:0.2f})'.format(labelname[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(task_name)
    plt.legend(loc="lower right", fontsize=8, prop=zhfont)
    plt.show()

    plt.savefig(args.init_from + '/roc.png')
    plt.savefig('figures/' + task_name + '.png')
    print(args.init_from + '/roc.png')
    print('figures/' + task_name + '.png')
plot()

#plt.plot([1,2,3])
#plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.xlabel('fds')
#plt.ylabel('fdsfds')
#plt.savefig('tmp.png')

