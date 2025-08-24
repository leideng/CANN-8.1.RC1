#!/bin/bash
# Uninstall the run package of aoe.
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

is_quiet=n
curpath=$(dirname $(readlink -f $0))
common_func_path="${curpath}/common_func.inc"
pkg_version_path="${curpath}/../version.info"
COMMON_SHELL="${curpath}/common.sh"
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE
docker_root_install_path=""

. "${common_func_path}"

# load shell
source "${COMMON_SHELL}"

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

AOE_INSTALL_DIR_PATH="${common_parse_dir}/${pkg_relative_path}"
WHL_LOCAL_DIR_PATH="${common_parse_dir}/${pkg_relative_path}/python/site-packages"

install_info="${common_parse_dir}/${pkg_relative_path}/ascend_install.info"
aoe_user_name=$(get_install_param "Aoe_UserName" "${install_info}")
aoe_user_group=$(get_install_param "Aoe_UserGroup" "${install_info}")
username=$aoe_user_name
usergroup=$aoe_user_group
if [ "$username" == "" ]; then
    username=$DEFAULT_USERNAME
    usergroup=$DEFAULT_USERGROUP
fi

sourcedir="${common_parse_dir}/${pkg_relative_path}"
SOURCE_INSTALL_COMMON_PARSER_FILE="${common_parse_dir}/${pkg_relative_path}/script/install_common_parser.sh"
SOURCE_FILELIST_FILE="${common_parse_dir}/${pkg_relative_path}/script/filelist.csv"

new_echo() {
    local log_type_=${1}
    local log_msg_=${2}
    if  [ "${is_quiet}" = "n" ]; then
        echo ${log_type_} ${log_msg_} 1 > /dev/null
    fi
}

##########################################################################
log "INFO" "step into run_aoe_uninstall.sh ......"
log "INFO" "uninstall targetdir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir/${pkg_relative_path}" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir/${pkg_relative_path} is not exist."
    exit 1
fi

### whl包卸载
uninstall_whl_package() {
    local _module="$1"
    local _module_apth="$2"
    if [ ! -d "${WHL_LOCAL_DIR_PATH}/${_module}" ]; then
        pip3 show ${_module} >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "${_module} is not exist."
        else
            pip3 uninstall -y "${_module}" 1> /dev/null
            if [ $? -ne 0 ]; then
                log "WARNING" "uninstall ${_module} failed."
                exit 1
            else
                log "INFO" "uninstall ${_module} successful."
            fi
        fi
    else
        export PYTHONPATH="${_module_apth}"
        pip3 uninstall -y "${_module}" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "uninstall ${_module} failed."
            exit 1
        else
            log "INFO" "uninstall ${_module} successful."
        fi
    fi
    rm -rf "${AOE_INSTALL_DIR_PATH}/python" >/dev/null 2>&1
}

new_uninstall() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to uninstall ${PACKAGE_NAME} files."
        return 0
    fi

    chmod +w -Rf "${SOURCE_INSTALL_COMMON_PARSER_FILE}"

    get_version "pkg_version" "$pkg_version_path"
    get_version_dir "pkg_version_dir" "$pkg_version_path"

    # 执行卸载
    bash "$SOURCE_INSTALL_COMMON_PARSER_FILE" --package="aoe" --uninstall --username="$username" \
        --usergroup="$usergroup" --recreate-softlink --version=$pkg_version --version-dir=$pkg_version_dir \
        --docker-root="${docker_root_install_path}" "$common_parse_type" "$input_install_dir" "$SOURCE_FILELIST_FILE"
    if [ $? -ne 0 ]; then
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
