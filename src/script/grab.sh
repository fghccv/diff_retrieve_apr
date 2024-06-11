#!/bin/bash
#SBATCH -J minicpm                           # 作业名为 test
#SBATCH -o /home/zhoushiqi/workplace/apr/src/script/log/grab.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /home/zhoushiqi/workplace/apr/src/script/log/grab.err
#SBATCH -p hit                           # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 200:00:00                            # 任务运行的最长时间为 1 小时                          
#SBATCH --gres=gpu:8

#SBATCH -c 128

# conda
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
# killall -u zhoushiqi
python /home/zhoushiqi/workplace/graduate_project/script/generate_sentence_replace_humaneval/grab.py