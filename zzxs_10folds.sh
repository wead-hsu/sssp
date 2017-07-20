save_dir=data/zhongao/proc/zzxs/070719.tag/cv_

for i in `seq 0 9`;
do
	CUDA_VISIBLE_DEVICES=0 nohup python3 clf_with_tag.py \
	    --train_path ${save_dir}${i}/train.data.idx \
	    --valid_path ${save_dir}${i}/valid.data.idx \
	    --test_path ${save_dir}${i}/test.data.idx \
	    --vocab_path ${save_dir}${i}/vocab.pkl \
	    --labels_path ${save_dir}${i}/class_map.pkl \
	    --save_dir results/zhongao/zzxs/gru-chartagcv-notag/${i} \
	    --classifier_type "GRU" \
	    --num_classes 8 \
	    --save_every -1 &\
	    #--use_tag \
	    #--fix_sent_len 300 \
	    #--num_filters 30 \
	    #--validate_every 100\
	    #--filter_size 6 \
	    #--fixirrelevant \
	    #--init_from results/tmp4/rnn_test-16000
done
