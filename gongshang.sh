dir="data/gongshang/clf/"
save_dir="results/gongshang/clf-gongshang-fixirrelevant"

for i in `seq 0 14`;
do
	CUDA_VISIBLE_DEVICES=1 nohup python3 clf.py \
		--model_path sssp.models.clf.clf_fixirrelevant \
		--rnn_type GatedGRU \
		--train_path ${dir}/labeled.data.idx \
		--valid_path ${dir}/valid.data.idx \
		--test_path ${dir}/test.data.idx \
		--vocab_path ${dir}/vocab.pkl \
		--save_dir ${save_dir}/$i \
		--vocab_size 4291 \
		--num_tasks 15\
		--list_num_classes "3,3,5,5,5,5,5,5,5,5,5,5,13,5,5" \
		--task_id $i  &
done
