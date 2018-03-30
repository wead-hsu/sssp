save_dir=data/google-extraction/proc/0.1

CUDA_VISIBLE_DEVICES=3 python3 semiclf.py \
	--model_path sssp.models.semiclf.semiclf_sample_aaai17 \
	--classifier_type 'GRU+selfatt' \
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--vocab_size 20000 \
	--num_pretrain_steps 8000 \
	--save_dir results/google-extraction/semiclf-selfatt-nouseweight-0.1 \
	--num_classes 5 \
	#--use_weights
