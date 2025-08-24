#!/bin/bash
# Perform upgrade for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME=$(id -un)
DEFAULT_USERGROUP==$(id -gn)
is_quiet=n
setenv_flag=n
docker_root=""
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE
package_name="ncs"
pkg_prefix_path="tools"
pkg_relative_path="${pkg_prefix_path}/${package_name}"
sourcedir=$PWD/$pkg_relative_path

curpath=$(dirname $(readlink -f $0))
common_func_path="${curpath}/common_func.inc"
common_func_v2_path="${curpath}/common_func_v2.inc"

. "${common_func_path}"
. "${common_func_v2_path}"

pkg_version_path="${curpath}/../../version.info"
common_shell="${curpath}/common.sh"
. ${pkg_version_path}
source ${common_shell}

if [ "$1" ];then
    input_install_dir="$2"
    common_parse_type=$3
    is_quiet=$4
    setenv_flag=$5
    is_docker_install="${6}"
    docker_root=$7
    in_install_for_all=$8
fi

common_parse_dir=$input_install_dir
if [ x"${docker_root}" != "x" ]; then
    common_parse_dir=${docker_root}${input_install_dir}
else
    common_parse_dir=${input_install_dir}
fi

is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ]; then
    get_version_dir "pkg_version_dir" "$pkg_version_path"
    common_parse_dir="$common_parse_dir/$pkg_version_dir"
fi

install_info="${common_parse_dir}/${pkg_relative_path}/ascend_install.info"

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
log_file="${log_dir}/ascend_install.log"

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
        echo ${log_type_} ${log_msg_} 1> /dev/null
    fi
}

if [ -f "$install_info" ]; then
    #. $install_info
    ncs_user_name=$(get_install_param "ncs_user_name" "${install_info}")
    ncs_user_group=$(get_install_param "ncs_user_group" "${install_info}")
    username=$ncs_user_name
    usergroup=$ncs_user_group
elif [ -f "$install_info_old" ] && [ $(grep -c -i "ncs_install_path_param" $install_info_old) -ne 0 ]; then
    username=$(grep -i username= "${install_info_old}" | cut -d"=" -f2-)
    usergroup=$(grep -i usergroup= "${install_info_old}" | cut -d"=" -f2-)
else
    echo "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete ${install_info} or ${install_info_old}"
    exit 1
fi

ncs_install_type=$(get_install_param "ncs_install_type" "${install_info}")
if [ "$username" == "" ]; then
    username=$DEFAULT_USERNAME
    usergroup=$DEFAULT_USERGROUP
fi

output_progress() {
    new_echo "INFO" "ncs upgrade upgradePercentage:$1%"
    log "INFO" "ncs upgrade upgradePercentage:$1%"
}

##########################################################################
log "INFO" "step into run_ncs_upgrade.sh ......"

log "INFO" "upgrade targetdir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

new_upgrade() {
    output_progress 10
    if [[ "${setenv_flag}" == "y" ]];then
        setenv_option="--setenv"
    else
        setenv_option=""
    fi

    get_version "pkg_version" "$pkg_version_path"
    get_version_dir "pkg_version_dir" "$pkg_version_path"

    # 执行安装
    custom_options="--custom-options=--stage=upgrade,--quiet=$is_quiet"
    bash "$curpath/install_common_parser.sh" --package=${package_name} --install --username="$username" \
        --usergroup="${usergroup}" --set-cann-uninstall --upgrade --version=$pkg_version \
        --version-dir=$pkg_version_dir ${setenv_option} ${in_install_for_all} \
        --docker-root="${docker_root}" ${custom_options} "${common_parse_type}" "${input_install_dir}" \
        "${curpath}/filelist.csv"
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to chown files."
        return 1
    fi

    return 0
}

new_upgrade
if [ $? -ne 0 ];then
    exit 1
fi

output_progress 100
exit 0
