#!/bin/bash
# Perform custom_install script for aoe package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

PACKAGE_NAME="aoe"
PACKAGE_PATH="tools"
curpath=$(dirname $(readlink -f $0))
USER_ASSETS_PATH="${curpath}/user_assets.inc"
common_func_v2_path="${curpath}/common_func_v2.inc"
version_cfg_path="${curpath}/version_cfg.inc"
COMMON_SHELL="${curpath}/common.sh"
is_migrate_user_assets_flag=y

. "${common_func_v2_path}"
. "${version_cfg_path}"

# load shell
source "${COMMON_SHELL}"

common_parse_dir=""
common_parse_type=""
stage=""

while true; do
    case "$1" in
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --common-parse-type=*)
        common_parse_type=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
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

migrate_user_assets() {
    export ASCEND_HOME=""

    local _default_install_path="/usr/local/Ascend/${pkg_relative_path}"
    if [ $(id -u) -ne 0 ]; then
        _default_install_path="${HOME}/Ascend/${pkg_relative_path}"
    fi

    source "${USER_ASSETS_PATH}"
    # search custom bank
    search_user_assets "ASCEND_HOME" "" "${_default_install_path}" data path_list
    if [ "x$path_list" != "x" ]; then
        OLD_IFS="$IFS"
        IFS=';'
        for src in $path_list
        do
            dst=${src##*${PACKAGE_NAME}/}
            dst=${dst%data}
            dst="$common_parse_dir/${pkg_relative_path}/$dst"
            if [ "${src}" != "${dst}data" ]; then
                log "INFO" "cp -rn $src $dst"
                cp -rn $src $dst
            fi
        done
        IFS="$OLD_IFS"
    fi

    # search aoe.ini and merge
    search_user_assets "ASCEND_HOME" "" "${_default_install_path}" "aoe.ini" path_list
    if [ "x$path_list" != "x" ]; then
        OLD_IFS="$IFS"
        IFS=';'
        for src in $path_list
        do
            dst=${src##*${PACKAGE_NAME}/}
            dst="$common_parse_dir/${pkg_relative_path}/$dst"
            if [ "${src}" != "${dst}" ]; then
                merge_config $src $dst
            fi
        done
        IFS="$OLD_IFS"
    fi
}

install_whl_package() {
    local _package_path="$1"
    local _package_name="$2"
    local _pythonlocalpath="$3"
    log "INFO" "start install python module package ${_package_name}."
    if [ -f "${_package_path}" ]; then
        pip3 install --upgrade --no-deps --force-reinstall "${_package_path}" -t "${_pythonlocalpath}" 1> /dev/null
        if [ $? -ne 0 ]; then
            log "WARNING" "install ${_package_name} failed."
            exit 1
        else
            log "INFO" "install ${_package_name} successful."
        fi
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:install ${_package_name} failed, can not find the matched package for this platform."
        exit 1
    fi
}


custom_install() {
    # merge config file
    merge_config "$common_parse_dir/${pkg_relative_path}/conf/aoe.ini" "${curpath}/../conf/aoe.ini"
    if [ $? -ne 0 ];then
        log "ERROR" "Merger aoe.ini failed."
    fi


    if [ "${is_migrate_user_assets_flag}" = y ]; then
        migrate_user_assets
        if [ $? -ne 0 ]; then
            log "WARNING" "failed to merge infos to custom directories"
        fi
    fi

    return 0
}

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
