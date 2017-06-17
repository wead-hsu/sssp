CUDA_VISIBLE_DEVICES=0 nohup python3 clf.py \
	--model_path sssp.models.clf.twostep_clf \
	--train_path data/zhongao/tasks_hasnan_hierachy/Code_Zasd_ybsd/train.data.idx \
	--train_label_path data/zhongao/tasks_hasnan_hierachy/Code_Zasd_ybsd/train.data.idx \
	--train_unlabel_path data/zhongao/tasks_hasnan_hierachy/Code_Zasd_ybsd/train.data.idx \
	--valid_path data/zhongao/tasks_hasnan_hierachy/Code_Zasd_ybsd/valid.data.idx \
	--test_path data/zhongao/tasks_hasnan_hierachy/Code_Zasd_ybsd/test.data.idx \
	--vocab_path data/zhongao/tasks_hasnan_hierachy/Code_Zasd_ybsd/vocab.pkl \
	--save_dir results/zhongao/twostep1-hasnan-hierachy/Code_Zasd_ybsd/ \
	--rnn_type GatedGRU \
	--vocab_size 15000 \
	--num_classes 16 &
	#--init_from results/zhongao/twostep-hasnan-hierachy/Code_Zasd_ybsd/rnn_test-8000 
