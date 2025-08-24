#!/bin/bash
# Perform install/upgrade/uninstall for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

package_name="ncs"
pkg_prefix_path="tools"
pkg_relative_path="${pkg_prefix_path}/${package_name}"

# 写日志
if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
logfile="${log_dir}/ascend_install.log"

log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_="${1}"
    local msg_="${2}"
    local log_format_="[Ncs] [$cur_date_] [$log_type_]: ${msg_}"
    if [ "$log_type_" == "INFO" ]; then
        echo "${log_format_}"
    elif [ "$log_type_" == "WARNING" ]; then
        echo "${log_format_}"
    elif [ "$log_type_" == "ERROR" ]; then
        echo "${log_format_}"
    elif [ "$log_type_" == "DEBUG" ]; then
        echo "${log_format_}" 1> /dev/null
    fi
    echo "${log_format_}" >> $logfile
}

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param_install

    if [ ! -f "${_file}" ];then
        exit 1
    fi
    local install_info_key_array=("ncs_install_type" "ncs_user_name" "ncs_user_group" "ncs_install_path_param")
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            local _param_install=$(grep -r "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param_install}"
}