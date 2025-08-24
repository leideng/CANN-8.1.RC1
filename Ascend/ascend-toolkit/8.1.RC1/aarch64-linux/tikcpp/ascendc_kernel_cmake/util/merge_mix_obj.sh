#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
# script for merge mix obj
# version: 1.0
# change log:
# ***********************************************************************
set -e

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
    --aic-dir)
        aic_dir=$2
        shift 2
        ;;
    --aiv-dir)
        aiv_dir=$2
        shift 2
        ;;
    --build-type)
        build_type=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

current_dir=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
mix_build_flag=mix_build.flag
aic_build_flag=aic_build.flag
aiv_build_flag=aiv_build.flag

if [ ! -d "${output}" ]; then
    mkdir -p ${output}
fi

if [ -n "${mix_build_flag}" ]; then
    rm -rf ${output}/${mix_build_flag}
fi

if [ -n "${aic_build_flag}" ]; then
    rm -rf ${output}/${aic_build_flag}
fi

if [ -n "${aiv_build_flag}" ]; then
    rm -rf ${output}/${aiv_build_flag}
fi

# mix mode
if [ -f "${aic_dir}/${mix_build_flag}" ] && [ -f "${aiv_dir}/${mix_build_flag}" ] ; then
    bash ${current_dir}/merge_obj.sh -l ${linker} -o ${output} -t ${build_type} -n device.o -m ${aic_dir}/device.o ${aiv_dir}/device.o

    touch ${output}/${mix_build_flag}
fi

# aic mode
if [ -f "${aic_dir}/${aic_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh -l ${linker} -o ${output} -t ${build_type} -n device_aic.o -m ${aic_dir}/device_aic.o

    touch ${output}/${aic_build_flag}
fi

# aiv mode
if [ -f "${aiv_dir}/${aiv_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh -l ${linker} -o ${output} -t ${build_type} -n device_aiv.o -m ${aiv_dir}/device_aiv.o

    touch ${output}/${aiv_build_flag}
fi
