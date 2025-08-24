#!/bin/bash
# The common processing function of the aoe installation script
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

PACKAGE_NAME="aoe"
PACKAGE_PATH="tools"
pkg_relative_path="${PACKAGE_PATH}/${PACKAGE_NAME}"
COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME="$(id -un)"
DEFAULT_USERGROUP="$(id -gn)"

install_info_key_array=("Aoe_Install_Type" "Aoe_UserName" "Aoe_UserGroup" "Aoe_Install_Path_Param")

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
log_file="${log_dir}/ascend_install.log"

update_install_param() {
    local _key=$1
    local _val=$2
    local _file=$3
    local _param

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            _param=$(grep -r "${_key}=" "${_file}")
            if [ "x${_param}" = "x" ]; then
                echo "${_key}=${_val}" >> "${_file}"
            else
                sed -i "/^${_key}=/c ${_key}=${_val}" "${_file}"
            fi
            break
        fi
    done
}

get_install_param() {
    local _key=$1
    local _file=$2
    local _param

    if [ ! -f "${_file}" ];then
        exit 1
    fi

    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            _param=$(grep -r "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

# 写日志
log() {
    local _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    local _log_type=$1
    local _msg="$2"
    local _log_format="[Aoe] [$_cur_date] [$_log_type]: ${_msg}"
    if [ "$_log_type" == "INFO" ]; then
        echo "${_log_format}"
    elif [ "$_log_type" == "WARNING" ]; then
        echo "${_log_format}"
    elif [ "$_log_type" == "ERROR" ]; then
        echo "${_log_format}"
    elif [ "$_log_type" == "DEBUG" ]; then
        echo "$_log_format" 1> /dev/null
    fi
    echo "${_log_format}" >> $log_file
}

merge_config() {
    local _old_file=$1
    local _new_file=$2
    local _tmp
    local is_modified_flag="false"
    log "INFO" "merge ${_old_file}, ${_new_file}"
    _old_file=$(realpath -q ${_old_file})
    _new_file=$(realpath -q ${_new_file})
    if [ "${_old_file}" == "${_new_file}" ]; then
        return 0
    fi

    if [ ! -f "${_old_file}" ] || [ ! -f "${_new_file}" ]; then
        return 0
    fi

    diff "${_old_file}" "${_new_file}" > /dev/null
    if [ $? -eq 0 ]; then
        return 0 # old file content equal new file content
    fi

    # check file permission
    if [ ! -r "${_old_file}" ] && [ ! -w "${_new_file}" ]; then
        return 1
    fi

    cp -f "${_old_file}" "${_new_file}.old"
    _old_file="${_new_file}.old"
    while read _line; do
        _tmp=$(echo "${_line}" | sed "s/ //g")
        if [ x"${_tmp}" == "x" ]; then
            continue # null line
        fi
        _tmp=$(echo "${_tmp}" | cut -d"#" -f1)
        if [ x"${_tmp}" == "x" ]; then
            continue # the line is comment
        fi

        _tmp=$(echo "${_line}" | grep "=")
        if [ x"${_tmp}" == "x" ]; then
            continue
        fi

        local _key=$(echo "${_line}" | cut -d"=" -f1)
        local _value=$(echo "${_line}" | cut -d"=" -f2-)
        if [ x"${_key}" == "x" ]; then
            echo "the config format is unrecognized, line=${_line}"
            continue
        fi
        if [ x"${_value}" == "x" ]; then
            continue
        fi
        # replace config value to new file
        if [[ "${is_modified_flag}" == "false" ]]; then
            log "WARNING" "${_new_file} has been modified!"
            is_modified_flag="true"
        fi
        sed -i "/^${_key}=/c ${_key}=${_value}" "${_new_file}"
    done < "${_old_file}"

    rm -f "${_old_file}"
}

get_dir_mod() {
    local path="$1"
    stat -c %a "$path"
}

remove_dir_recursive() {
    local dir_start="$1"
    local dir_end="$2"
    if [ "$dir_end" = "$dir_start" ]; then
        return 0
    fi
    if [ ! -e "$dir_end" ]; then
        return 0
    fi
    if [ "x$(ls -A $dir_end 2>&1)" != "x" ]; then
        return 0
    fi
    local up_dir="$(dirname $dir_end)"
    local oldmod="$(get_dir_mod $up_dir)"
    chmod u+w "$up_dir"
    [ -n "$dir_end" ] && rm -rf "$dir_end"
    if [ $? -ne 0 ]; then
        chmod "$oldmod" "$up_dir"
        return 1
    fi
    chmod "$oldmod" "$up_dir"
    remove_dir_recursive "$dir_start" "$up_dir"
}
