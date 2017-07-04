save_dir=data/zhongao/tasks_hasnan/Code_Zasd_ybsd/

CUDA_VISIBLE_DEVICES=1 python3 clf.py \
	--model_path sssp.models.clf.basic_clf \
	--train_path ${save_dir}/train.data.idx \
	--train_label_path ${save_dir}/train.data.idx \
	--train_unlabel_path ${save_dir}/train.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--save_dir results/tmp \
	--rnn_type "Tag=bigru;Gatedgru=bwctx+tag" \
	--fixirrelevant True \
	--num_classes 17
	#--init_from results/tmp4/rnn_test-16000
