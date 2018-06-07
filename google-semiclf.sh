save_dir=data/google-extraction/proc/0.2

CUDA_VISIBLE_DEVICES=2 python3 semiclf.py \
	--model_path sssp.models.semiclf.semiclf_aaai17 \
	--classifier_type 'GRU+selfatt' \
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--vocab_size 20000 \
	--num_pretrain_steps 000 \
	--num_classes 5 \
	--max_sent_len 200 \
	--save_dir results/google-extraction/semiclf-bnfix-selfatt-useweight-0.2-maxsentlen200-pretrain0 \
	--use_weights \
	#--use_binaryweights \
	#--decoder_type 'sclstm'\
	#--save_dir results/tmp \
