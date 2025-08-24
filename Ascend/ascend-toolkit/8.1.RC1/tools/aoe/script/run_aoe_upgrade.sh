#!/bin/bash
# Upgrade the run package of aoe
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

is_quiet=n
input_setenv=n
sourcedir=$PWD/${PACKAGE_NAME}
curpath=$(dirname $(readlink -f $0))
common_func_path="${curpath}/common_func.inc"
pkg_version_path="${curpath}/../../version.info"
COMMON_SHELL="${curpath}/common.sh"
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE
input_install_for_all=n
in_install_for_all=""
docker_root_install_path=""

. "${common_func_path}"

# load shell
source "${COMMON_SHELL}"

if [ "$1" ];then
    input_install_dir="$2"
    common_parse_type=$3
    is_quiet=$4
    input_install_for_all=$5
    input_setenv=$6
    docker_root_install_path=$7
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

log "INFO" "${PACKAGE_NAME} install for all ${input_install_for_all}"
if [ "${input_install_for_all}" = "y" ]; then
    in_install_for_all="--install_for_all"
    log "INFO" "${PACKAGE_NAME} enable install for all"
fi

install_info="${common_parse_dir}/${pkg_relative_path}/ascend_install.info"
install_info_old="/etc/ascend_install.info"

new_echo() {
    local log_type_=${1}
    local log_msg_=${2}
    if  [ "${is_quiet}" = "n" ]; then
        echo ${log_type_} ${log_msg_} 1> /dev/null
    fi
}

if [ -f "$install_info" ]; then
    #. $install_info
    aoe_user_name=$(get_install_param "Aoe_UserName" "${install_info}")
    aoe_user_group=$(get_install_param "Aoe_UserGroup" "${install_info}")
    username=$aoe_user_name
    usergroup=$aoe_user_group
elif [ -f "$install_info_old" ] && [ $(grep -c -i "Aoe_Install_Path_Param" "$install_info_old") -ne 0 ]; then
    username=$(grep -i username= "${install_info_old}" | cut -d"=" -f2-)
    usergroup=$(grep -i usergroup= "${install_info_old}" | cut -d"=" -f2-)
else
    echo "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete ${install_info} or ${install_info_old}"
    exit 1
fi

aoe_install_type=$(get_install_param "Aoe_Install_Type" "${install_info}")
if [ "$username" == "" ]; then
    username=$DEFAULT_USERNAME
    usergroup=$DEFAULT_USERGROUP
fi

output_progress() {
    new_echo "INFO" "${PACKAGE_NAME} upgrade upgradePercentage:$1%"
    log "INFO" "${PACKAGE_NAME} upgrade upgradePercentage:$1%"
}

##########################################################################
log "INFO" "step into run_aoe_upgrade.sh ......"

log "INFO" "upgrade targetdir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

new_upgrade() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to upgrade ${PACKAGE_NAME} files."
        return 0
    fi
    output_progress 10

    local setenv_option=""
    if [ "${input_setenv}" = y ]; then
        setenv_option="--setenv"
    fi

    get_version "pkg_version" "$pkg_version_path"
    get_version_dir "pkg_version_dir" "$pkg_version_path"

    # 执行安装
    custom_options="--custom-options=--common-parse-dir=${common_parse_dir},--common-parse-type=${common_parse_type},--stage=upgrade"
    bash "$curpath/install_common_parser.sh" --package="aoe" --install --username="$username" --usergroup="$usergroup" --set-cann-uninstall --upgrade \
        --version=$pkg_version --version-dir=$pkg_version_dir \
        $setenv_option $in_install_for_all --docker-root="$docker_root_install_path" $custom_options "$common_parse_type" "$input_install_dir" "$curpath/filelist.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to install package."
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
