save_dir=data/zhongao/proc/zzxs/070704/inchar_cv

for i in `seq 0 9`;
do
	CUDA_VISIBLE_DEVICES=0 python3 clf.py \
	    --model_path sssp.models.clf.basic_clf \
	    --train_path ${save_dir}${i}/train.data.idx \
	    --train_label_path ${save_dir}${i}/train.data.idx \
	    --train_unlabel_path ${save_dir}${i}/train.data.idx \
	    --valid_path ${save_dir}${i}/valid.data.idx \
	    --test_path ${save_dir}${i}/test.data.idx \
	    --vocab_path ${save_dir}${i}/vocab.pkl \
	    --labels_path ${save_dir}${i}/class_map.pkl \
	    --save_dir results/zhongao/zzxs/cnn3layer-numfilter_30-char-cv/${i} \
	    --encoder_type "CNN3layer" \
	    --num_classes 8 \
	    --validate_every 100\
	    --max_epoch 100 \
	    --save_every -1 \
	    --fix_sent_len 300 \
	    --num_filters 30 \
	    #--filter_size 6 \
	    #--fixirrelevant \
	    #--init_from results/tmp4/rnn_test-16000
done
