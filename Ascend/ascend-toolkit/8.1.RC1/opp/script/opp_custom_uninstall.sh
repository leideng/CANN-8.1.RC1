#!/bin/bash
# Perform custom remove softlink script for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_COMMON="${curpath}""/opp_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${OPP_COMMON}"
. "${common_func_path}"
# init arch 
architecture=$(uname -m)
architectureDir="${architecture}-linux"

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done
get_version_dir "opp_kernel_version_dir" "$install_path/$version_dir/opp_kernel/version.info"

if [ -z "$opp_kernel_version_dir" ]; then
    # before remove the oppkernel, remove the softlinks
    logandprint "[INFO]: Start remove opapi softlinks."
    softlinksRemove ${install_path}/${version_dir}
    if [ $? -ne 0 ]; then
        logandprint "[WARNING]: Remove opapi softlinks failed, some softlinks may not exist."
    else
        logandprint "[INFO]: Remove opapi softlinks successfully."
    fi
fi
