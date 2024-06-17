#!/bin/bash
#SBATCH -J minicpm                           # 作业名为 test
#SBATCH -o /home/zhoushiqi/workplace/apr/src/script/log/evaluate.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /home/zhoushiqi/workplace/apr/src/script/log/evaluate.err
#SBATCH -p hit                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 200:00:00                            # 任务运行的最长时间为 1 小时                          
#SBATCH --gres=gpu:8
#SBATCH --mem 500GB
#SBATCH -c 128

# conda
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
model=/home/zhoushiqi/workplace/model/codet5_commit
codebase_path=/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process_filtered2048.jsonl
dest_dir=/home/zhoushiqi/workplace/apr/data/vectors
mkdir -p $dest_dir/temp
index=0
gpu_num=8

for ((i = 0; i < $gpu_num; i++)); do

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python /home/zhoushiqi/workplace/apr/src/code/retrival/vector_generate.py --model_id ${model} \
      --gpu_index ${i} \
      --dest_path ${dest_dir}/temp/${i}.jsonl \
      --codebase_path ${codebase_path} 
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
echo merge
python /home/zhoushiqi/workplace/apr/src/code/merge.py --merge_dir $dest_dir/temp --dest_path $dest_dir/all_vector_codet5_commit.jsonl
