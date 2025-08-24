#!/bin/bash
# Perform uninstall for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME=$(id -un)
DEFAULT_USERGROUP=$(id -gn)
is_quiet=n
common_parse_type=$COMMON_INSTALL_TYPE
docker_root=""

curpath=$(dirname $(readlink -f $0))
common_func_path="${curpath}/common_func.inc"
common_func_v2_path="${curpath}/common_func_v2.inc"

. "${common_func_path}"
. "${common_func_v2_path}"

pkg_version_path="${curpath}/../version.info"
common_shell="${curpath}/common.sh"
. ${pkg_version_path}
source ${common_shell}

if [ "$1" ];then
    input_install_dir="$2"
    common_parse_type=$3
    is_quiet=$4
    docker_root_install_path=$5
fi

if [ "x${docker_root_install_path}" != "x" ]; then
    common_parse_dir="${docker_root_install_path}${input_install_dir}"
else
    common_parse_dir="${input_install_dir}"
fi

is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ]; then
    get_version_dir "pkg_version_dir" "$pkg_version_path"
    common_parse_dir="$common_parse_dir/$pkg_version_dir"
fi

install_info="${common_parse_dir}/${pkg_relative_path}/ascend_install.info"
sourcedir="${common_parse_dir}/${pkg_relative_path}"

ncs_user_name=$(get_install_param "ncs_user_name" "${install_info}")
ncs_user_group=$(get_install_param "ncs_user_group" "${install_info}")
username=$ncs_user_name
usergroup=$ncs_user_group
if [ "$username" == "" ]; then
    username=$DEFAULT_USERNAME
    usergroup=$DEFAULT_USERGROUP
fi

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

log_file="${log_dir}/ascend_install.log"

SOURCE_PATH="${common_parse_dir}/${pkg_relative_path}"
SOURCE_INSTALL_COMMON_PARSER_FILE="${common_parse_dir}/${pkg_relative_path}/script/install_common_parser.sh"
SOURCE_FILELIST_FILE="${common_parse_dir}/${pkg_relative_path}/script/filelist.csv"

log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_=$1
    local msg_="$2"
    local log_format_="[Ncs] [$cur_date_] [$log_type_]: ${msg_}"
    if [ "$log_type_" == "INFO" ]; then
        echo "${log_format_}"
    elif [ "$log_type_" == "WARNING" ]; then
        echo "${log_format_}"
    elif [ "$log_type_" == "ERROR" ]; then
        echo "${log_format_}"
    elif [ "$log_type_" == "DEBUG" ]; then
        echo "$log_format_" 1> /dev/null
    fi
    echo "${log_format_}" >> $log_file
}

new_echo() {
    local log_type_=${1}
    local log_msg_=${2}
    if  [ "${is_quiet}" = "n" ]; then
        echo ${log_type_} ${log_msg_} 1 > /dev/null
    fi
}

log "INFO" "step into run_ncs_uninstall.sh ......"

log "INFO" "uninstall targetdir $common_parse_dir , type $common_parse_type."

if [ ! -d "$common_parse_dir/${pkg_relative_path}" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir/${pkg_relative_path} is not exist."
    exit 1
fi

new_uninstall() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to uninstall ncs files."
        return 0
    fi

    get_version "pkg_version" "$pkg_version_path"
    get_version_dir "pkg_version_dir" "$pkg_version_path"

    bash "$SOURCE_INSTALL_COMMON_PARSER_FILE" --package="ncs" --uninstall --username="$username" \
        --usergroup="$usergroup" --recreate-softlink --version=$pkg_version --version-dir=$pkg_version_dir \
        --docker-root="${docker_root_install_path}" "$common_parse_type" "$input_install_dir" "$SOURCE_FILELIST_FILE"

    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0090;ERR_DES:failed to uninstall package."
        return 1
    fi

    return 0
}

new_uninstall

if [ $? -ne 0 ];then
    exit 1
fi

exit 0
