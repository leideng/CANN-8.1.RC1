#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
# script for merge obj
# version: 1.0
# change log:
# ***********************************************************************
set -e

while [[ $# -gt 0 ]]; do
    case $1 in
    -s)
        src=$2
        shift 2
        ;;
    -d)
        dst=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

if [  -n "${dst}" ]; then
    rm -rf ${dst}
fi

if [ ! -d "${dst}" ]; then
    mkdir -p ${dst}
fi

for arg in "$@"
do
    relative_file=$(realpath --relative-to="${src}" "${arg}")
    dst_file=${dst}/${relative_file}
    dst_dir=$(dirname ${dst_file})
    if [ ! -d "${dst_dir}" ]; then
        mkdir -p ${dst_dir}
    fi
    cp -v ${arg} ${dst_dir}
done

