declare -A nc     # Explicitly declare
declare -A vs     # Explicitly declare

nc[Code_Zasd_pp]=5
vs[Code_Zasd_pp]=15000
nc[Code_Zasd_ybsd]=17
vs[Code_Zasd_ybsd]=15000
nc[Code_Zasd_xp]=5
vs[Code_Zasd_xp]=15000
nc[Code_Zasd_qb]=3
vs[Code_Zasd_qb]=15000
nc[Code_Zasd_blsd]=6
vs[Code_Zasd_blsd]=15000
nc[Code_Zasd_cc]=6
vs[Code_Zasd_cc]=15000
nc[Code_Zasd_hsqq]=5
vs[Code_Zasd_hsqq]=15000
nc[Code_Zasd_zd]=2
vs[Code_Zasd_zd]=15000
nc[Code_Zasd_cm]=4
vs[Code_Zasd_cm]=15000
nc[Code_zatd_stfs]=5
vs[Code_zatd_stfs]=15000
nc[Code_zatd_xwtd]=32
vs[Code_zatd_xwtd]=15000
nc[Code_zatd_zzxs]=5
vs[Code_zatd_zzxs]=15000
nc[Code_zatd_zafw]=4
vs[Code_zatd_zafw]=15000

#echo "First Method: ${nc[Code_Zasd_pp]}"
#echo "Second Method: ${nc[@]}"

#echo "${nc[Code_Zasd_blsd]}"


dir="data/zhongao/tasks_hasnan/"
save_dir="results/zhongao/taggatedgru-hasnan"
for f in $(ls ${dir});
do
	echo "CUDA_VISIBLE_DEVICES=1 nohup python3 clf.py --rnn_type tagGatedGRU \
		--train_path ${dir}/$f/train.data.idx \
		--train_label_path ${dir}/$f/train.data.idx \
		--train_unlabel_path ${dir}/$f/train.data.idx \
		--valid_path ${dir}/$f/valid.data.idx \
		--test_path ${dir}/$f/test.data.idx \
		--vocab_path ${dir}/$f/vocab.pkl \
		--save_dir ${save_dir}/$f \
		--vocab_size ${vs[${f}]} \
		--num_classes ${nc[${f}]} &"

	CUDA_VISIBLE_DEVICES=1 nohup python3 clf.py --rnn_type tagGatedGRU \
		--train_path ${dir}/$f/train.data.idx \
		--train_label_path ${dir}/$f/train.data.idx \
		--train_unlabel_path ${dir}/$f/train.data.idx \
		--valid_path ${dir}/$f/valid.data.idx \
		--test_path ${dir}/$f/test.data.idx \
		--vocab_path ${dir}/$f/vocab.pkl \
		--save_dir ${save_dir}/$f \
		--vocab_size ${vs[${f}]} \
		--num_classes ${nc[${f}]} &
done
