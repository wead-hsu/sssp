save_dir='data/agnews/ag8000'

CUDA_VISIBLE_DEVICES=3 python3 semiclf.py \
	--model_path sssp.models.semiclf.ssvae_tf \
	--classifier_type 'LSTM' \
	--batch_size_label 5\
	--batch_size_unlabel 55\
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--embd_path ${save_dir}/embd.pkl \
	--save_dir 'results/ssvae_tf_0605_mod1/' \
	--num_classes 4 \
	--num_pretrain_steps 0000 \
	--vocab_size 23829 \
	--sample_unlabel 'False' \
	--klw_b 120000 \
	--klw_w 0.000052 \
	#--use_weights
