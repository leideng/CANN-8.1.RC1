#!/bin/bash
USERNAME=$(id -un)
USERGROUP=$(id -gn)

SHELL_DIR=$(cd "$(dirname "$0")" || exit; pwd)
COMMON_PARSER_PATH=${SHELL_DIR}"/parser_install.sh"
COMMON_SHELL_PATH=${SHELL_DIR}"/common.sh"
FILELIST_CSV_PATH=${SHELL_DIR}"/filelist.csv"


function print_usage() {
    local _ret=$1

    echo "Usage: $0 [Options]"
    echo "Options:"
    echo "    --help | -h   : Print out this help message"
    echo "    --quiet       : Quiet uninstall mode, skip human-computer interactions"
    exit ${_ret}
}

function real_install_path() {
    if [ ! -f ${install_file} ]; then
        log_and_print $LEVEL_WARN "Install file ${install_file} doesn't exist."
        return 1
    fi
    install_path=$(get_install_param "Install_Path_Param" ${install_file})
    if [ -z "${install_path}" ]; then
        log_and_print $LEVEL_WARN "Install path is empty from 'ascend_install.info'."
        return 1
    fi
    install_path=$(cd "${install_path}" && pwd)
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_WARN "Install path ${install_path} doesn't exist which read from 'ascend_install.info'."
        return 1
    fi
    local _path=$(dirname ${install_path})
    latest_path=${_path}"/latest"
    return 0
}

function check_dir_permission() {
    local _path=$1

    if [ -z ${_path} ]; then
        log_and_print $LEVEL_ERROR "The dir path is empty, uninstall failed."
        exit_log 1
    fi
    if [ ! -d "${_path}" ]; then
        log_and_print $LEVEL_ERROR "The dir path ${_path} does not exist, uninstall failed."
        exit_log 1
    fi
    if [ "$(id -u)" -eq 0 ]; then
        return
    fi
    if [ ! -r ${_path} ] || [ ! -w ${_path} ] || [ ! -x ${_path} ]; then
        log_and_print $LEVEL_ERROR "The user $USERNAME should have read, write and executable permission for ${_path}."
        exit_log 1
    fi
}

function restoremod_file() {
    if [ -d ${install_path}/python/site-packages/mspti-0.0.1.dist-info/ ]; then
        chmod u+w ${install_path}/python/site-packages/mspti-0.0.1.dist-info/
    fi
}

function uninstall_tool() {
    restoremod_file
    whlUninstallPackage mspti ${install_path}/python/site-packages
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Remove mindstudio-toolkit mspti whl failed in ${install_path}."
        return 1
    fi
    # when normal user uninstall package, shell need to restore dir permission
    "$COMMON_PARSER_PATH" --restoremod --package=mindstudio-toolkit --username="unknown" --usergroup="unknown" \
        "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Restore directory written permission failed."
        return 1
    fi
    "$COMMON_PARSER_PATH" --remove --package=mindstudio-toolkit "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "ERR_NO:0X0090;ERR_DES: Remove mindstudio-toolkit files failed in ${install_path}."
        return 1
    fi
    log $LEVEL_INFO "Remove mindstudio-toolkit files succeed in ${install_path}!"
    return 0
}

function uninstall() {
    install_file=$(cd ${SHELL_DIR}"/.." && pwd)"/ascend_install.info"

    real_install_path
    [ $? -ne 0 ] && return 1
    check_dir_permission ${install_path}
    check_dir_permission ${install_path}"/mindstudio-toolkit"
    should_latest_uninstall ${latest_path}
    local _is_latest_uninstall=$?
    if [ ${_is_latest_uninstall} -eq 0 ]; then
        uninstall_latest ${latest_path}
        if [ $? -ne 0 ]; then
            log_and_print ${LEVEL_ERROR} "Mindstudio toolkit uninstall lastest failed, please retry."
            return 1
        fi
    fi

    uninstall_tool
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "Mindstudio toolkit uninstall failed."
        return 1
    fi
    rm -f ${install_file}
    unregister_uninstall ${install_path}
    remove_uninstall_file_if_no_content ${install_path}"/cann_uninstall.sh"
    remove_empty_dir ${install_path}"/mindstudio-toolkit"
    if [ ${_is_latest_uninstall} -eq 0 ]; then
        switch_to_the_previous_version ${latest_path}
    fi
    remove_empty_dir "${install_path}/${ARCH}-${OS}/include"
    remove_empty_dir ${install_path}
    remove_empty_dir $(dirname ${install_path})
    log_and_print $LEVEL_INFO "Mindstudio toolkit uninstall success!"
}

source ${COMMON_SHELL_PATH}

install_file=""
quiet_flag=n

start_log

log_and_print $LEVEL_INFO "LogFile: $log_file"
log_and_print $LEVEL_INFO "InputParams: $*"

while true; do
    case "$1" in
    --quiet)
        quiet_flag=y
        shift
        ;;
    --help | -h)
        print_usage 0
        shift
        ;;
    -*)
        echo "Unsupported parameters: $1"
        print_usage 1
        ;;
    *)
        break
        ;;
    esac
done

uninstall
exit_log 0
