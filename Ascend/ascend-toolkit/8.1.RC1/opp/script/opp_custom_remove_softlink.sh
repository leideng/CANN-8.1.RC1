#!/bin/bash
# Perform custom remove softlink script for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_COMMON="${curpath}""/opp_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${OPP_COMMON}"
. "${common_func_path}"

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
    latestSoftlinksRemove "$install_path/$version_dir"
fi

# if opp exists, restore the softlink of the opp version
get_version_dir "kernel_latest_dir" "${install_path}/$latest_dir/opp_kernel/version.info"

if [ -n "$kernel_latest_dir" ]; then
    createOpapiLatestSoftlink ${install_path}/${kernel_latest_dir} opp_kernel
fi

level2_dir="$install_path/$latest_dir/include/aclnnop/level2"
if [ -d "$level2_dir" ] && [ "$(ls -A "$level2_dir")" = "" ]; then
    rm -rf "$level2_dir"
fi
 
aclnnop_dir="$install_path/$latest_dir/include/aclnnop"
if [ -d "$aclnnop_dir" ] && [ "$(ls -A "$aclnnop_dir")" = "" ]; then
    rm -rf "$aclnnop_dir"
fi
 
rm -rf "$install_path/$latest_dir/ops"
rm -rf "$install_path/$latest_dir/opp"
