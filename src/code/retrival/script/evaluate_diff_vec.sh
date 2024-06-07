#!/bin/bash
#SBATCH -J minicpm                           # 作业名为 test
#SBATCH -o /home/zhoushiqi/workplace/apr/src/script/log/evaluate.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /home/zhoushiqi/workplace/apr/src/script/log/evaluate.err
#SBATCH -p gpu4                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 200:00:00                            # 任务运行的最长时间为 1 小时                          
#SBATCH --gres=gpu:8
#SBATCH --mem 500GB
#SBATCH -c 128
#SBATCH -w g4002
# conda
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
model=/home/zhoushiqi/workplace/model/deepseek-coder-6.7b-instruct
work_dir=/home/zhoushiqi/workplace/apr/data/evaluate_results/deepseek/retrieval
pname=baseline_1.2_diff_vec_k2
info_path=/home/zhoushiqi/workplace/apr/data/df4_process_data/one_function/1.2.jsonl
result_path=/home/zhoushiqi/workplace/apr/data/evaluate_results/deepseek/retrieval/baseline_1.2_random_N200_T1_result.jsonl
process_path=/home/zhoushiqi/workplace/apr/data/evaluate_results/deepseek/retrieval/baseline_1.2_random_N200_T1_process.jsonl
codebase_path=/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process_filtered2048.jsonl
vector_path="/home/zhoushiqi/workplace/apr/data/vectors/all_vector_2048.jsonl"
num_example=2
num_ticket=3
num_voter=100
N=100
num_per_iter=10
T=1

pname=${pname}_N${N}_T${T}
dest_dir=$work_dir/$pname
rm -rf $dest_dir
mkdir -p $dest_dir
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python /home/zhoushiqi/workplace/apr/src/code/retrival/evaluate_diff_vector.py --model_id ${model} \
      --gpu_index ${i}  --num_gpus 1\
      --dest_path ${dest_dir}/${i}.jsonl --N $N --num_per_iter $num_per_iter --temperature $T\
      --codebase_path ${codebase_path} --num_example ${num_example} --num_ticket ${num_ticket}\
      --info_path $info_path --result_path $result_path --process_path $process_path --num_voter $num_voter
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
echo merge
python /home/zhoushiqi/workplace/apr/src/code/merge.py --merge_dir $dest_dir --dest_path $work_dir/$pname.jsonl
echo process
python /home/zhoushiqi/workplace/apr/src/code/process.py --data_path $work_dir/$pname.jsonl --dest_path $work_dir/${pname}_process.jsonl
echo run test
python /home/zhoushiqi/workplace/apr/src/code/replace_evaluate.py --version 1.7 --infos_path $info_path --process_path $work_dir/${pname}_process.jsonl\
                                                                    --result_path  $work_dir/${pname}_result.jsonl