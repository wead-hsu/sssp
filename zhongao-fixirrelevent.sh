declare -A nc     # Explicitly declare
declare -A vs     # Explicitly declare
#
#nc[Code_Zasd_pp]=4
#vs[Code_Zasd_pp]=15000
#nc[Code_Zasd_ybsd]=16
#vs[Code_Zasd_ybsd]=15000
#nc[Code_Zasd_xp]=4
#vs[Code_Zasd_xp]=15000
#nc[Code_Zasd_qb]=2
#vs[Code_Zasd_qb]=15000
#nc[Code_Zasd_blsd]=5
#vs[Code_Zasd_blsd]=15000
#nc[Code_Zasd_cc]=5
#vs[Code_Zasd_cc]=15000
#nc[Code_Zasd_hsqq]=4
#vs[Code_Zasd_hsqq]=15000
#nc[Code_Zasd_zd]=1
#vs[Code_Zasd_zd]=15000
#nc[Code_Zasd_cm]=3
#vs[Code_Zasd_cm]=15000
#nc[Code_zatd_stfs]=4
#vs[Code_zatd_stfs]=15000
#nc[Code_zatd_xwtd]=31
#vs[Code_zatd_xwtd]=15000
#nc[Code_zatd_zzxs]=4
#vs[Code_zatd_zzxs]=15000
#nc[Code_zatd_zafw]=3
#vs[Code_zatd_zafw]=15000

nc[Code_Zasd_ybsd]=17
vs[Code_Zasd_ybsd]=15000
nc[Code_Zasd_qb]=3
vs[Code_Zasd_qb]=15000
nc[Code_Zasd_cm]=4
vs[Code_Zasd_cm]=15000
nc[Code_zatd_xwtd]=32
vs[Code_zatd_xwtd]=15000
nc[Code_zatd_zzxs]=5
vs[Code_zatd_zzxs]=15000

#echo "First Method: ${nc[Code_Zasd_pp]}"
#echo "Second Method: ${nc[@]}"

#echo "${nc[Code_Zasd_blsd]}"


data_dir="data/zhongao/proc/070606/tasks_hasnan_inchar"
save_dir="results/zhongao/gru-hasnan-inchar-fixirrelevant"
for f in $(ls ${data_dir});
do
	CUDA_VISIBLE_DEVICES=1 nohup python3 clf.py \
		--model_path sssp.models.clf.clf_fixirrelevant \
		--rnn_type GRU \
		--train_path ${data_dir}/$f/train.data.idx \
		--train_label_path ${data_dir}/$f/train.data.idx \
		--train_unlabel_path ${data_dir}/$f/train.data.idx \
		--valid_path ${data_dir}/$f/valid.data.idx \
		--test_path ${data_dir}/$f/test.data.idx \
		--vocab_path ${data_dir}/$f/vocab.pkl \
		--save_dir ${save_dir}/$f \
		--num_classes ${nc[${f}]} &
		#--vocab_size ${vs[${f}]} 
done