save_dir=data/case_type_clf/proc/

CUDA_VISIBLE_DEVICES=0 python3 semiclf.py \
	--model_path sssp.models.semiclf.semiclf_sample_aaai17 \
	--classifier_type 'GatedGRU' \
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--save_dir results/case/semiclf-gatedgru-sample-useweightforclf/ \
	--num_classes 12 \
	--use_weights
