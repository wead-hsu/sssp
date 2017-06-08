import sys
import os
import time
from sssp.utils import utils
from sssp.config import exp_logger
from sssp.io.datasets import initDataset
from sssp.io.batch_iterator import threaded_generator
from sssp.utils.utils import average_res, res_to_string
import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG)

# ------------- CHANGE CONFIGURATIONS HERE ---------------
conf_dirs = ['sssp.config.conf_clf',]
#conf_dirs = ['sssp.config.conf_clf_multilabel']
# --------------------------------------------------------

def validate(valid_dset, model, sess):
    res_list = []
    threaded_it = threaded_generator(valid_dset, 200)
    for batch in threaded_it:
        res_dict, res_str, summary = model.run_batch(sess, batch, istrn=False)
        res_list.append(res_dict)
    out_str = res_to_string(average_res(res_list))
    return out_str

def train_and_validate(args, model, sess, train_dset, valid_dset, test_dset, explogger):
    batch_cnt = 0
    res_list = []

    # init tensorboard writer
    tb_writer = tf.summary.FileWriter(args.save_dir + '/train', sess.graph)

    t_time = time.time()
    time_cnt = 0
    for epoch_idx in range(args.max_epoch):
        threaded_it = threaded_generator(train_dset, 200)
        for batch in threaded_it:
            batch_cnt += 1
            gen_time = time.time() - t_time
            t_time = time.time()
            res_dict, res_str, summary = model.run_batch(sess, batch, istrn=True)
            run_time = time.time() - t_time
            res_dict.update({'run_time': run_time})
            res_dict.update({'gen_time': gen_time})
            res_list.append(res_dict)
            res_list = res_list[-200:]
            time_cnt += gen_time + run_time

            if batch_cnt % args.show_every == 0:
                tb_writer.add_summary(summary, batch_cnt)
                out_str = res_to_string(average_res(res_list))
                explogger.message(out_str, True)
            
            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = validate(valid_dset, model, sess)
                explogger.message('VALIDATE: '  + out_str, True)

            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = validate(test_dset, model, sess)
                explogger.message('TEST: ' + out_str, True)

            if batch_cnt % args.save_every == 0:
                save_fn = os.path.join(args.save_dir, args.log_prefix)
                explogger.message("Saving checkpoint model: {} ......".format(args.save_dir))
                model.saver.save(sess, save_fn, 
                        write_meta_graph=False,
                        global_step=batch_cnt)

            t_time = time.time()

def main():
    # load all args for the experiment
    args = utils.load_argparse_args(conf_dirs=conf_dirs)
    explogger = exp_logger.ExpLogger(args.log_prefix, args.save_dir)
    wargs = vars(args)
    wargs['conf_dirs'] = conf_dirs
    explogger.write_args(wargs)
    explogger.file_copy(['sssp'])

    # step 1: import specified model
    module = __import__(args.model_path, fromlist=[args.model_name])
    model_class = module.__dict__[args.model_name]
    model = model_class(args)
    vt, vs = model.model_setup(args)
    explogger.write_variables(vs)

    # step 2: init dataset
    train_dset = initDataset(args.train_path, model.get_prepare_func(args), args.batch_size)
    valid_dset = initDataset(args.valid_path, model.get_prepare_func(args), 1)
    test_dset = initDataset(args.test_path, model.get_prepare_func(args), 1)

    # step 3: Init tensorflow
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        if args.init_from:
            model.saver.restore(sess, args.init_from)
            explogger.message('Model restored from {0}'.format(args.init_from))
        else:
            tf.global_variables_initializer().run()
        train_and_validate(args, 
                model=model, 
                sess=sess, 
                train_dset=train_dset,
                valid_dset=valid_dset,
                test_dset=test_dset,
                explogger=explogger)
