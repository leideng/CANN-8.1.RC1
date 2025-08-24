#!/bin/bash
# Perform custom remove softlink script for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_KERNEL_COMMON="${curpath}""/opp_kernel_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${common_func_path}"
. "${OPP_KERNEL_COMMON}"
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

latest_softlinks_remove "$install_path/$version_dir"

rm -rf "$install_path/$latest_dir/opp_latest"

# if opp exists, restore the softlink of the opp version
get_version_dir "opp_version_dir" "${install_path}/$latest_dir/opp/version.info"

if [ -n "$opp_version_dir" ]; then
    create_opapi_latest_softlink ${install_path}/${opp_version_dir} opp
fi