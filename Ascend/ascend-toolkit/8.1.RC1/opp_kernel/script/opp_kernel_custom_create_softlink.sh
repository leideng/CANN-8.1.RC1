#!/bin/bash
# Perform custom create softlink script for compiler package
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

# create op_api soft link
logPrint "[INFO]: Start create opapi latest softlinks."
create_opapi_latest_softlink ${install_path}/${version_dir}
return_code=$?
if [ ${return_code} -eq 0 ]; then
    logPrint "[Info]: Create opapi latest softlinks successfully!"
elif [ ${return_code} -eq 3 ]; then
    logPrint "[WARNING]: opapi source file does not exist!"
else
    logPrint "[ERROR]: Create opapi latest softlinks failed!"
fi

get_version_dir "latest_version_dir" "$install_path/$latest_dir/runtime/version.info"
 
if [ -d "$install_path/$version_dir/opp" ] && [ -n "$latest_version_dir" ] && [ -d "$install_path/$latest_version_dir" ]; then
    if [ "${latest_version_dir}" != "${version_dir}" ]; then
        logPrint "[Info]: latest version is $latest_version_dir, the pkg version is $version_dir, and the opp_latest soft link is created."
        ln -srfn "$install_path/$version_dir/opp" "$install_path/$latest_dir/opp_latest"
    elif [ "${latest_version_dir}" == "${version_dir}" ]; then
        rm -rf "$install_path/$latest_dir/opp_latest"
    fi
fi

