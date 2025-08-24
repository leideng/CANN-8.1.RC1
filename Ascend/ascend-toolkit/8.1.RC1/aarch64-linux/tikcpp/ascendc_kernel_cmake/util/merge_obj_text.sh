#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# script for inplace merge obj text sections for template kernels
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
    *)
        break
        ;;
    esac
done

for arg in "$@"
do
    echo -n "${linker} -m aicorelinux -Ttext=0 ${arg} -o ${arg}"
    # template kernel text sections will be named as .text.<mangle> like .text._Z11hello_worldILi300EhEvT0_f
    # to remain consistent with none template kernels, merge them into .text
    ${linker} -m aicorelinux -Ttext=0 ${arg} -o ${arg}
done
