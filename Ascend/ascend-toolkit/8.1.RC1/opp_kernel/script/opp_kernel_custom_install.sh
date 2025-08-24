#!/bin/sh
# Perform custom_install script for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

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


custom_install() {
    # create op_api soft link
    logPrint "[INFO]: Start create opapi softlinks."
    create_opapi_softlink "${install_path}/${version_dir}"
    return_code=$?
    if [ ${return_code} -eq 0 ]; then
        logPrint "[Info]: Create opapi softlinks successfully!"
    elif [ ${return_code} -eq 3 ]; then
        logPrint "[WARNING]: opapi source file does not exist!"
    else
        logPrint "[ERROR]: Create opapi softlinks failed!"
        exit 1
    fi

    return 0
}

runtime_version_info="${install_path}/latest/runtime/version.info"
kernel_version_info="${install_path}/${version_dir}/opp_kernel/version.info"
 
get_version "kernel_version" ${kernel_version_info}
kernel_short_version=$(echo "${kernel_version}" | cut -d '.' -f 1,2)
if [ -f ${runtime_version_info} ]; then
    current_version=$(cat "${runtime_version_info}" | grep "^required_opp_abi_version" | cut -d= -f2- | tr -d '"')
    if [ -z "${current_version}" ] || [ -z "${kernel_short_version}" ]; then
        logPrint "[Info]: version is empty"
    elif echo "${current_version}" | grep -q "<=7.6"; then
        original_permissions=$(stat -c "%a" "${runtime_version_info}")
        chmod u+w "${runtime_version_info}"
        sed -i "/^required_opp_abi_version/ s/<=7.6/<=7.7/" "${runtime_version_info}"
        logPrint "[Info]: required_opp_abi_version support has been updated to include ${kernel_short_version}"
        chmod "${original_permissions}" "${runtime_version_info}"
    else
        logPrint "[Info]: No changes needed for required_opp_abi_version: ${current_version}"
    fi
fi

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0