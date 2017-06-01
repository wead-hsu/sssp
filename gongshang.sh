for i in `seq 0 15`;
do
	 CUDA_VISIBLE_DEVICES=0 nohup python3 run.py --save_dir results/gongshang/clf-gongshang-gatedctxgru2-tasks/$i --task_id $i  &> ~/tmp/$i.log
done
