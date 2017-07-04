import argparse

def init_arguments(parser):
    # MODEL
    parser.add_argument('--model_path', type=str, default='sssp.models.clf.clf_fixirrelevant', help='model_path')
    parser.add_argument('--model_name', type=str, default='RnnClassifier', help='model_name')
    parser.add_argument('--rnn_type', type=str, default='GatedGRU', help='Type of RNN')
    parser.add_argument('--num_units', type=int, default=512, help='Dimension of hidden state of RNN')
    parser.add_argument('--batch_size', type=int, default=25, help='batch_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--fixirrelevant', type=bool, default=False, help='If fix the NAN logit')
    parser.add_argument('--w_regl1', type=float, default=0.0, help='coefficient for l1 regulariztion')
    parser.add_argument('--w_regdiff', type=float, default=0.0, help='coefficient for l1 diff regulariztion')
    parser.add_argument('--w_regsharp', type=float, default=0.0, help='coefficient for l1 diff regulariztion')

    # TRAINING
    parser.add_argument('--max_epoch', type=int, default=4000, help='Maximum number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.0, help='grad_clip')
    parser.add_argument('--max_norm', type=float, default=20.0, help='max_norm')
    parser.add_argument('--train_embd', type=bool, default=True, help='If train embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--show_every', type=int, default=100, help='Number of batch between showing the results')
    parser.add_argument('--save_every', type=int, default=2000, help='Number of batch between saving the results')
    parser.add_argument('--validate_every', type=int, default=1000, help='Number of batch between validating the results')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay_rate')
    parser.add_argument('--decay_steps', type=float, default=100, help='decay_steps')
    parser.add_argument('--keep_rate', type=float, default=0.2, help='keep_rate')
    parser.add_argument('--labels_path', type=str, default=None, help='label names')

    # DATASET
    parser.add_argument('--list_num_classes', type=str, default='', help='list of number of classes')
    parser.add_argument('--num_tasks', type=int, default=None, help='Number of tasks')
    parser.add_argument('--task_id', type=int, default=None, help='id of the task')

    parser.add_argument('--train_path', type=str, default='data/case_type_clf/proc/train_all.data.idx', help='Directory of datasets')
    parser.add_argument('--train_label_path', type=str, default='data/case_type_clf/proc/train_all.data.idx', help='Directory of datasets')
    parser.add_argument('--train_unlabel_path', type=str, default='data/case_type_clf/proc/unlabeled.data.idx', help='Directory of datasets')
    parser.add_argument('--valid_path', type=str, default='data/case_type_clf/proc/dev.data.idx', help='Directory of datasets')
    parser.add_argument('--test_path', type=str, default='data/case_type_clf/proc/test.data.idx', help='Directory of datasets')
    parser.add_argument('--vocab_path', type=str, default='data/case_type_clf/proc/vocab.pkl', help='vocab_path')
    parser.add_argument('--save_dir', type=str, default='results/case/clf_gatedgru-noforget', help='Directory for saving')
    parser.add_argument('--init_from', type=str, default=None, help='Restore from the trained model path')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Size of vocabulary')
    parser.add_argument('--embd_path', type=str, default=None, help='')
    parser.add_argument('--max_sent_len', type=int, default=100, help='maximum sentence length')
   
    # ENVORIMENTS
    parser.add_argument('--embd_dim', type=int, default=300, help='Dimension of embedding matrix')
    parser.add_argument('--max_to_keep', type=int, default=2, help='max_to_keep')
    parser.add_argument('--log_prefix', type=str, default='rnn_test', help='Log prefix')
