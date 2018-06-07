#save_dir=/home/wdxu/codes/ssvae-wdxu-mod/scripts/examples/tmp/
save_dir=data/google-extraction/proc/0.4

CUDA_VISIBLE_DEVICES=3 python3 clf.py \
	--model_path sssp.models.clf.basic_clf \
	--train_path ${save_dir}/labeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--save_dir results/google-extraction/clf-0.4-lstm-maxlen200 \
	--classifier_type "LSTM" \
	--num_classes 5 \
	--vocab_size 20000\
	--validate_every 1000 \
	--max_sent_len 200 \
	--log_prefix 'clf' \
	--embd_dim 300 \
	#--keep_rate 0.8 \
	#--num_filters 150 \
	#--fix_sent_len 400 \
	#--embd_path ${save_dir}/embd.pkl \
	#--num_units 512 \
	#--max_norm 0.0 \
	#--max_sent_len 100 \
	#--vocab_path ${save_dir}/vocab.pkl \
	#--fix_sent_len 300 \
	#--labels_path ${save_dir}/class_map.pkl \
	#--num_filters 300 \
	#--w_regfrobenius 0.001 \
	#--w_regdiff 0.00003\
	#--w_regl1 0.00006\
	#--filter_size 3 \
	#--batch_size 64 \
	#--fixirrelevant \
	#--init_from results/tmp4/rnn_test-16000

