save_dir=data/zhongao/proc/zzxs/070719.tag/

CUDA_VISIBLE_DEVICES=1 python3 clf_with_tag.py \
	--train_path ${save_dir}/train.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--labels_path ${save_dir}/class_map.pkl \
	--save_dir results/tmp1 \
	--classifier_type "CNN+multiscale" \
	--num_classes 8 \
	--fix_sent_len 300 \
	--validate_every 100 \
	#--use_tag \
	#--w_regdiff 0.00003\
	#--w_regl1 0.00006\
	#--num_filters 60 \
	#--filter_size 3 \
	#--batch_size 64 \
	#--fixirrelevant \
	#--init_from results/tmp4/rnn_test-16000
