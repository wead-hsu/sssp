save_dir=data/imdb/imdb20000/

CUDA_VISIBLE_DEVICES=1 python3 semiclf.py \
	--model_path sssp.models.semiclf.semiclf_sample_aaai17 \
	--classifier_type 'LSTM' \
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--embd_path ${save_dir}/embd.pkl \
	--save_dir results/imdb/semiclf-imdb20000-lstm-sample-useweight/ \
	--vocab_size 20000 \
	--num_classes 2 \
	--num_pretrain_steps 40000 \
	--use_weights
