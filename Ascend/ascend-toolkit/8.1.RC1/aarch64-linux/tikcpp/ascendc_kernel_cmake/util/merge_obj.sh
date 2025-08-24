#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
# script for merge obj
# version: 1.0
# change log:
# ***********************************************************************
set -e

build_type="Debug"
merge_flag=false

while [[ $# -gt 0 ]]; do
    case $1 in
    -l | --linker)
        linker=$2
        shift 2
        ;;
    -o | --output)
        output=$2
        shift 2
        ;;
    -t)
        build_type=$2
        shift 2
        ;;
    -m)
        merge_flag=true
        shift
        ;;
    -n)
        name=$2
        shift 2
        ;;
    -f)
        flag=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

if [ ! -f "${linker}" ]; then
    echo "error: ${linker} does not exist."
    exit 1
fi

if [ ! -d "${output}" ]; then
    mkdir -p ${output}
fi

# Do not set the -r compiler option when generating the final device.o
if [ "${merge_flag}" == "false" ]; then
    r_option="-r"
else
    n_option=""
fi

# generate the final device.o, and set the -x compiler option for Release
if [ "${merge_flag}" == "true" ] && [ "${build_type}" == "Release" ]; then
    x_option="-x"
fi

echo "${linker} ${x_option} -m aicorelinux ${r_option} ${n_option} -Ttext=0 $@ -static -o ${output}/${name}"
${linker} ${x_option} -m aicorelinux ${r_option} ${n_option} -Ttext=0 $@ -static -o ${output}/${name}

# generate obj file list
if [ -n "${flag}" ]; then
    flag_dir=$(dirname ${flag})

    if [ -f "${flag}" ]; then
        rm -f ${flag}
    fi

    if [ ! -d "${flag_dir}" ]; then
        mkdir -p ${flag_dir}
    fi

    for arg in "$@"
    do
        echo "$arg" >> ${flag}
    done
fi
