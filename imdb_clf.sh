save_dir=/home/wdxu/codes/ssvae-wdxu-mod/scripts/examples/tmp/
#save_dir=data/imdb/imdb20000/

CUDA_VISIBLE_DEVICES=3 python3 clf.py \
	--model_path sssp.models.clf.basic_clf \
	--train_path ${save_dir}/train.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--save_dir results/tmp \
	--classifier_type "LSTM" \
	--num_classes 2 \
	--vocab_size 20000\
	--validate_every 4000 \
	--max_norm 0.0 \
	--embd_path ${save_dir}/embd.pkl \
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
