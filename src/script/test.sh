#!/bin/bash
dirs=(/home/zhoushiqi/workplace/apr/df4/defects4j-2.0.1/framework/projects/*/)
for dir in "${dirs[@]}"; do
    echo $(basename "${dir%/}")  # 移除尾部的斜线
done
