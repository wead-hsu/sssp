save_dir=data/zhongao/proc/zzxs/070704/inchar/

CUDA_VISIBLE_DEVICES=1 python3 clf.py \
	--model_path sssp.models.clf.basic_clf \
	--train_path ${save_dir}/train.data.idx \
	--train_label_path ${save_dir}/train.data.idx \
	--train_unlabel_path ${save_dir}/train.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--labels_path ${save_dir}/class_map.pkl \
	--save_dir results/zhongao/zzxs/cnn-char-filtersize_6-numfilters_30 \
	--encoder_type "CNN" \
	--num_classes 8 \
	--fix_sent_len 300 \
	--num_filters 30 \
	--filter_size 6 \
	#--fixirrelevant \
	#--init_from results/tmp4/rnn_test-16000
