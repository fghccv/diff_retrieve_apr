#!/bin/bash
. "/home/zhoushiqi/anaconda3/etc/profile.d/conda.sh"
conda activate apr
ori_dir=$1
PATH=/home/zhoushiqi/perl5/bin:/home/zhoushiqi/anaconda3/envs/apr/bin:/data/apps/tools/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/opt/slurm/bin:/opt/slurm/bin
export PATH="$PATH:$ori_dir/framework/bin"
# defects4j info -p Lang
if [ $(basename $ori_dir) == "defects4j-2.0.1" ]
then 
  version=1.8
else 
  version=1.7
fi
path=/home/zhoushiqi/lib/jvm/jdk$version
echo java=$version
export JAVA_HOME=$path
export PATH=$JAVA_HOME/bin:$PATH
# java -version
num=0
dirs=($ori_dir/framework/projects/*/)
declare -A dict4_1p2

dict4_1p2["Chart"]=$(seq 1 26)
dict4_1p2["Closure"]=$(seq 1 133)
dict4_1p2["Lang"]=$(seq 1 65)
dict4_1p2["Math"]=$(seq 1 106)
dict4_1p2["Mockito"]=$(seq 1 38)
dict4_1p2["Time"]=$(seq 1 27)



for dir in "${dirs[@]}"; do
    project=$(basename "${dir%/}")  # 假设每个目录是一个项目
    
    if [ "$project" == "lib" ]; then
        continue
    fi
    echo $project 
    if [ $(basename $ori_dir) == "defects4j-2.0.1" ]
    then 
        ids=$(defects4j bids -p $project)
    else 
        ids=${dict4_1p2["$project"]}
    fi 
    for id in $ids; do  # 这里不需要 ${ids[*]}，直接使用 $ids
        num=$((num + 1))
        echo $id 
        stor_dir=/home/zhoushiqi/workplace/apr/df4/all_project/$(basename $ori_dir)/$project/$id
        rm -rf $stor_dir
        mkdir -p $stor_dir
        defects4j checkout -p $project -v ${id}b -w $stor_dir
    done
done
echo "All projects count: $num"

    

