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
    --pack_tool)
        pack_tool=$2
        shift 2
        ;;
    --elf_in)
        elf_in=$2
        shift 2
        ;;
    --add_dir)
        add_dir=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

mix_build_flag=mix_build.flag
aiv_build_flag=aiv_build.flag
aic_build_flag=aic_build.flag

# mix mode
if [ -f "${add_dir}/${mix_build_flag}" ]; then
    echo "${pack_tool} ${elf_in} ${add_dir}/device.o 0 ${elf_in}"
    ${pack_tool} ${elf_in} ${add_dir}/device.o 0 ${elf_in}
fi

# aiv mode
if [ -f "${add_dir}/${aiv_build_flag}" ]; then
    echo "${pack_tool} ${elf_in} ${add_dir}/device_aiv.o 1 ${elf_in}"
    ${pack_tool} ${elf_in} ${add_dir}/device_aiv.o 1 ${elf_in}
fi

# aic mode
if [ -f "${add_dir}/${aic_build_flag}" ]; then
    echo "${pack_tool} ${elf_in} ${add_dir}/device_aic.o 2 ${elf_in}"
    ${pack_tool} ${elf_in} ${add_dir}/device_aic.o 2 ${elf_in}
fi
