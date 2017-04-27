# load difference flags configurations from 
# different files

import tensorflow as tf
import numpy as np
import argparse

tf.app.flags.DEFINE_string('dummy_constant', '-', '')

def load_tensorflow_args(conf_dirs):
    for cdir in conf_dirs:
        __import__(cdir)

    flags = tf.app.flags.FLAGS
    # Hard code here:
    # 1. get one arbitrary attribute in the flags
    #   to let the parser parse the arguments
    flags.dummy_constant
    return flags

def load_argparse_args(conf_dirs):
    parser = argparse.ArgumentParser()
    for cdir in conf_dirs:
        module = __import__(cdir, fromlist=['init_arguments'])
        f = module.__dict__['init_arguments']
        f(parser)
    
    flags = parser.parse_args()
    return flags

def res_to_string(res_dict):
    res_str = ''
    for k in sorted(res_dict.keys()):
        res_str += '{0}: {1:0.3f}'.format(k, res_dict[k]) + '\t'
    return res_str

def average_res(res_list):
    if len(res_list) == 0:
        return {}
    keys = res_list[0].keys()
    avg = {}
    for k in keys:
        avg[k] = np.mean([res[k] for res in res_list])
    return avg
