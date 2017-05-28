for i in `seq 11 15`;
do
	 CUDA_VISIBLE_DEVICES=0 nohup python3 run.py --save_dir results/clf-gongshang-task$i --task_id $i  &> ~/tmp/$i.log
done
