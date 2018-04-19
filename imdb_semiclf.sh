save_dir=data/imdb/imdb20000/

CUDA_VISIBLE_DEVICES=2 python3 semiclf.py \
	--model_path sssp.models.semiclf.semiclf_aaai17 \
	--sample_unlabel 'S1' \
	--classifier_type 'LSTM' \
	--num_units 256 \
	--keep_rate 0.95\
	--max_sent_len 400 \
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--embd_path ${save_dir}/embd.pkl \
	--save_dir results/imdb/semiclf-bnfix-imdb20000-lstm-256-0.5-maxlen400-s1-noweight \
	--vocab_size 20000 \
	--num_classes 2 \
	--num_pretrain_steps 8000 \
	--learning_rate 0.0004 \
	#--use_weights
	#--batch_size_label 25 \
	#--max_norm 0.0 \
