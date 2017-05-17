import argparse

def init_arguments(parser):
    # MODEL
    parser.add_argument('--model_path', type=str, default='sssp.models.semiclf.semiclf_gateatt', help='model_path')
    parser.add_argument('--model_name', type=str, default='SemiClassifier', help='model_name')
    parser.add_argument('--rnn_type', type=str, default='GRU', help='Type of RNN')
    parser.add_argument('--num_units', type=int, default=256, help='Dimension of hidden state of RNN')
    parser.add_argument('--batch_size_label', type=int, default=20, help='batch_size')
    parser.add_argument('--batch_size_unlabel', type=int, default=20, help='batch_size')
    parser.add_argument('--use_sampled_softmax', default=False, help='If use sampled_softmax to speed up')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of samples used in sampled_softmax')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--dim_z', type=int, default=100, help='Dimension of latent code')
    parser.add_argument('--alpha', type=float, default=1.0, help='rescale for unlabeled clf')
    parser.add_argument('--num_pretrain_steps', type=int, default=80000, help='Number of step for pretraining')

    # TRAINING
    parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='grad_clip')
    parser.add_argument('--max_norm', type=float, default=200, help='max_norm')
    parser.add_argument('--train_embd', type=bool, default=True, help='If train embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--show_every', type=int, default=100, help='Number of batch between showing the results')
    parser.add_argument('--save_every', type=int, default=2000, help='Number of batch between saving the results')
    parser.add_argument('--validate_every', type=int, default=1000, help='Number of batch between validating the results')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay_rate')
    parser.add_argument('--decay_steps', type=float, default=100, help='decay_steps')

    # DATASET
    dataset = 'case' #['beer', 'case']
    if dataset == 'case':
        parser.add_argument('--train_label_path', type=str, default='data/case_type_clf/proc/train_all.data.idx', help='Directory of datasets')
        parser.add_argument('--train_unlabel_path', type=str, default='data/case_type_clf/proc/unlabeled.data.idx', help='Directory of datasets')
        parser.add_argument('--valid_path', type=str, default='data/case_type_clf/proc/dev.data.idx', help='Directory of datasets')
        parser.add_argument('--test_path', type=str, default='data/case_type_clf/proc/test.data.idx', help='Directory of datasets')
        parser.add_argument('--vocab_path', type=str, default='data/case_type_clf/proc/vocab.pkl', help='vocab_path')
        parser.add_argument('--save_dir', type=str, default='results/semiclf-gru-constdecweight-all', help='Directory for saving')
        parser.add_argument('--klw_w', type=float, default=3e-5, help='klw = klw_w * step + klw_b')
        parser.add_argument('--klw_b', type=float, default=3e5, help='klw = klw_w * step + klw_b')
        #parser.add_argument('--init_from', type=str, default='results/semiclf/semiclf-8000', help='Restore from the trained model path')
        parser.add_argument('--init_from', type=str, default=None, help='Restore from the trained model path')
        parser.add_argument('--num_classes', type=int, default=12, help='Number of classes')
    elif dataset == 'beer':
        parser.add_argument('--train_label_path', type=str, default='data/beer/proc/cls_0-aspect_1-clf/train_all.data.idx', help='Directory of datasets')
        parser.add_argument('--train_unlabel_path', type=str, default='data/beer/proc/cls_0-aspect_1-clf/unlabeled.data.idx', help='Directory of datasets')
        parser.add_argument('--valid_path', type=str, default='data/beer/proc/cls_0-aspect_1-clf/dev.data.idx', help='Directory of datasets')
        parser.add_argument('--test_path', type=str, default='data/beer/proc/cls_0-aspect_1-clf/test.data.idx', help='Directory of datasets')
        parser.add_argument('--vocab_path', type=str, default='data/beer/proc/cls_0-aspect_1-clf/proc/vocab.pkl', help='vocab_path')
        parser.add_argument('--save_dir', type=str, default='results/tmp', help='Directory for saving')
        parser.add_argument('--klw_w', type=float, default=3e-5, help='klw = klw_w * step + klw_b')
        parser.add_argument('--klw_b', type=float, default=3e5, help='klw = klw_w * step + klw_b')
        #parser.add_argument('--init_from', type=str, default='results/semiclf/semiclf-8000', help='Restore from the trained model path')
        parser.add_argument('--init_from', type=str, default=None, help='Restore from the trained model path')
        parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    
    # ENVORIMENTS
    parser.add_argument('--vocab_size', type=int, default=20000, help='Size of vocabulary')
    parser.add_argument('--embd_path', type=str, default=None, help='embd_path')
    parser.add_argument('--embd_dim', type=int, default=300, help='Dimension of embedding matrix')
    parser.add_argument('--max_to_keep', type=int, default=2, help='max_to_keep')
    parser.add_argument('--log_prefix', type=str, default='semiclf', help='Log prefix')
