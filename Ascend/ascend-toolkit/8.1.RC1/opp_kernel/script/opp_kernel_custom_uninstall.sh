#!/bin/sh
# Perform custom uninstall script for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
OPP_KERNEL_COMMON="${curpath}""/opp_kernel_common.sh"
. "${common_func_path}"
. "${OPP_KERNEL_COMMON}"
common_parse_dir=""
logfile=""
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
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
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

custom_uninstall() {
    # before remove the oppkernel, remove the softlinks
    logPrint "[INFO]: Start remove opapi softlinks."
    softlinks_remove ${install_path}/${version_dir}
    if [ $? -ne 0 ]; then
        logPrint "[WARNING]: Remove opapi softlinks failed, some softlinks may not exist."
    else
        logPrint "[INFO]: Remove opapi softlinks successfully."
    fi
}

custom_uninstall

# if opp exists, restore the softlink of the opp version
get_version_dir "opp_version_dir" "${install_path}/$version_dir/opp/version.info"

if [ -n "$opp_version_dir" ]; then
    create_opapi_softlink ${install_path}/${opp_version_dir} opp
fi

