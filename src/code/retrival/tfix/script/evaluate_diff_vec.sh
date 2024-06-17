#!/bin/bash
#SBATCH -J minicpm                           # 作业名为 test
#SBATCH -o /home/zhoushiqi/workplace/apr/src/code/retrival/script/log/evaluate2.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /home/zhoushiqi/workplace/apr/src/code/retrival/script/log/evaluate2.err
#SBATCH -p hit                    # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 200:00:00                            # 任务运行的最长时间为 1 小时                          
#SBATCH --gres=gpu:8
#SBATCH --mem 500GB
#SBATCH -c 128

# conda
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
model=/home/zhoushiqi/workplace/model/deepseek-coder-6.7b-instruct
work_dir=/home/zhoushiqi/workplace/apr/data/evaluate_results/retrieval/deepseek/tfix
pname=diff_vec_codet5p_round1_random
result_path=/home/zhoushiqi/workplace/apr/data/evaluate_results/retrieval/deepseek/tfix/random_CaderPrompt_sm_N10_T1_k1_result.jsonl
codebase_path=/home/zhoushiqi/workplace/apr/data/tfix/data/train_data_clean_sm.json
vector_path=/home/zhoushiqi/workplace/apr/data/vectors/tfix_clean_sm_codet5p.jsonl
embedding_model=/home/zhoushiqi/workplace/model/codet5p-110m-embedding
data_path=/home/zhoushiqi/workplace/apr/data/tfix/data/test_data_clean_sm.json
num_example=1
num_ticket=3
num_voter=2
N=3
num_per_iter=3
T=1
uniform_weight=0
pname=${pname}_N${N}_T${T}_k${num_example}
dest_dir=$work_dir/$pname
# rm -rf $dest_dir
# mkdir -p $dest_dir
# index=0
# gpu_num=8
# for ((i = 0; i < $gpu_num; i++)); do
#   gpu=$((i))
#   ((index++))
#   (
#     CUDA_VISIBLE_DEVICES=$gpu python /home/zhoushiqi/workplace/apr/src/code/retrival/tfix/evaluate_diff_vector.py --model_id ${model} \
#       --gpu_index ${i}  --num_gpus 1\
#       --dest_path ${dest_dir}/${i}.jsonl --N $N --num_per_iter $num_per_iter --temperature $T\
#       --codebase_path ${codebase_path} --num_example ${num_example} --num_ticket ${num_ticket}\
#       --result_path $result_path  --num_voter $num_voter --uniform_weight $uniform_weight\
#       --vector_path $vector_path --embedding_model ${embedding_model} --data_path ${data_path}
#   ) &
#   if (($index % $gpu_num == 0)); then wait; fi
# done
echo merge
python /home/zhoushiqi/workplace/apr/src/code/merge.py --merge_dir $dest_dir --dest_path $work_dir/$pname.jsonl
echo process
python /home/zhoushiqi/workplace/apr/src/code/retrival/tfix/process.py --data_path $work_dir/$pname.jsonl --dest_path $work_dir/${pname}_process.jsonl
echo run test
python /home/zhoushiqi/workplace/apr/src/code/retrival/tfix/metric.py --data_path $work_dir/${pname}_process.jsonl --dest_path $work_dir/${pname}_result.jsonl \
                                                                      --metric em

# bash /home/zhoushiqi/workplace/apr/src/code/retrival/tfix/script/evaluate_diff_vec2.sh

