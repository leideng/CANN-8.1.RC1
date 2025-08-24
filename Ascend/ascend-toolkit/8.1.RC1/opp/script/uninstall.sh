#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2010-2018. All rights reserved.

_CURR_PATH=$(dirname $(readlink -f $0))

# error number and description
FILE_NOT_EXIST="0x0080"
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."
# log functions
getdate() {
    _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${_cur_date}"
}

logandprint() {
    is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
    if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
        echo "[Opp] [$(getdate)] ""$1"
    fi
    echo "[Opp] [$(getdate)] ""$1" >> "${_INSTALL_LOG_FILE}"
}

if [ "$(id -u)" != "0" ]; then
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
fi

# init install cmd status, set default as n
is_quiet=n
quiet_parameter=""
if [ "$#" != "0" ]; then
    if [ "$1" = "--quiet" ] && [ "$#" = "1" ]; then
        is_quiet=y
        quiet_parameter="--quiet"
    else
        logandprint "Please use correct parameters, only support input nothing or only --quiet parameter."
        exit 1
    fi
fi

install_shell="${_CURR_PATH}/install.sh"

# shell exist check
if [ ! -f "${install_shell}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};Opp module is not installed or some opp source files are lost.\
If there are any residual files, please manually remove those files."
    exit 1
fi

# shell execute perm check
if [ ! -x "${install_shell}" ]; then
    logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The user do \
not have the permission to execute this file, please reset the file \
to a right permission."
    exit 1
fi

installed_path="$(cd "${_CURR_PATH}/../../"; pwd)"
parent_installed_path="$(cd "${installed_path}/../"; pwd)"
cd ~
sh "${install_shell}" "--aa" "--aa" "--uninstall" "--install-path=${installed_path}" "${quiet_parameter}"
ret_status="$?"
if [ "${ret_status}" != "0" ]; then
    exit 1
fi
if [ -d "${parent_installed_path}" ];then
    subdirs_param_install=$(ls "${parent_installed_path}" 2> /dev/null)
    if [ "${subdirs_param_install}" = "" ]; then
        rm -rf "${parent_installed_path}"
    fi
fi
exit 0
