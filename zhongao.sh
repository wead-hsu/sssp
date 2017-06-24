declare -A nc     # Explicitly declare
declare -A vs     # Explicitly declare

nc[Code_Zasd_pp]=4
vs[Code_Zasd_pp]=798
nc[Code_Zasd_ybsd]=16
vs[Code_Zasd_ybsd]=10000
nc[Code_Zasd_xp]=4
vs[Code_Zasd_xp]=712
nc[Code_Zasd_qb]=2
vs[Code_Zasd_qb]=2957
nc[Code_Zasd_blsd]=5
vs[Code_Zasd_blsd]=941
nc[Code_Zasd_cc]=5
vs[Code_Zasd_cc]=1329
nc[Code_Zasd_hsqq]=4
vs[Code_Zasd_hsqq]=894
nc[Code_Zasd_zd]=1
vs[Code_Zasd_zd]=36
nc[Code_Zasd_cm]=3
vs[Code_Zasd_cm]=2418
nc[Code_zatd_stfs]=4
vs[Code_zatd_stfs]=965
nc[Code_zatd_xwtd]=31
vs[Code_zatd_xwtd]=10000
nc[Code_zatd_zzxs]=4
vs[Code_zatd_zzxs]=10000
nc[Code_zatd_zafw]=3
vs[Code_zatd_zafw]=1760

#echo "First Method: ${nc[Code_Zasd_pp]}"
#echo "Second Method: ${nc[@]}"

#echo "${nc[Code_Zasd_blsd]}"


dir="data/zhongao/proc/070616/tasks_hasnan/"
save_dir="results/zhongao/0616/gru-hasnan"
for f in $(ls ${dir});
do
	CUDA_VISIBLE_DEVICES=1 nohup python3 clf.py 
		--model_path sssp.models.clf.basic_clf\
		--rnn_type GRU \
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
