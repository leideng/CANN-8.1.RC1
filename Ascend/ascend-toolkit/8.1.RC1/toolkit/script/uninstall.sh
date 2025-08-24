#!/bin/bash
INSTALL_TYPE_ALL=full # run/devel/docker
PACKAGE_NAME=toolkit
LEVEL_INFO="INFO"
LEVEL_WARN="WARNING"
LEVEL_ERROR="ERROR"

INSTALL_LOG_FILE=ascend_install.log
SHELL_DIR=$(readlink -f $(cd "$(dirname "$0")" || exit;pwd))
UNINSTALL_SHELL="$SHELL_DIR/run_toolkit_uninstall.sh"
COMMON_SHELL="$SHELL_DIR/common.sh"
COMMON_INC="$SHELL_DIR/common_func.inc"
INSTALL_INFO_FILE=$(cd "$SHELL_DIR/../" || exit;pwd)/ascend_install.info
LOG_RELATIVE_PATH=/var/log/ascend_seclog # install log path and operation log path
VERSION_PATH="$SHELL_DIR/../version.info"

OPERATION_LOG_FILE=operation.log
LOG_OPERATION_UNINSTALL="Uninstall"
LOG_LEVEL_MAJOR="MAJOR"
LOG_RESULT_SUCCESS="success"
LOG_RESULT_FAILED="failed"
OPERATE_ADDR="127.0.0.1"

if [ "-$USER" = "-" ]; then
    # there is no USER in docker
    export USER=$(id -un)
fi

export docker_flag=n
export docker_root_path=""
export input_install_path=""

log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Toolkit] [${cur_date}] [$1]: $content" >> "${logFile}"
}

log_and_print() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Toolkit] [${cur_date}] [$1]: $content"
    echo "[Toolkit] [${cur_date}] [$1]: $content" >> "${logFile}"
}

print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Toolkit] [${cur_date}] [$1]: $content"
}

logOperation() {
    local operation=$1
    local errCode=$2
    local cmdList
    local result=$LOG_RESULT_SUCCESS
    local level=$LOG_LEVEL_MAJOR

    cmdList=`echo "$@" | cut -d" " -f3-`

    if [ $errCode -ne 0 ]; then
        result=$LOG_RESULT_FAILED
    fi

    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "${operation} ${level} ${USER} ${cur_date} ${OPERATE_ADDR} ${PACKAGE_NAME} ${result}; cmdlist=$cmdList" >> "${optLogFile}"
}

# check path permission
checkDirPermission() {
    local path=$1

    if [ x"${path}" = "x" ]; then
        log_and_print $LEVEL_ERROR "dir path ${path} is empty."
        return 1
    fi
    if [ ! -d "${path}" ]; then
        log_and_print $LEVEL_ERROR "dir path does not exist."
        return 1
    fi
    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi
    if [ ! -r "${path}" ] || [ ! -w "${path}" ] || [ ! -x "${path}" ]; then
        log_and_print $LEVEL_ERROR "The user $USER do not have the permission to access ${path}."
        return 1
    fi
    return 0
}

initLog() {
    local _log_path=${LOG_RELATIVE_PATH}
    local _cur_user=${USER}
    local _cur_group=`groups | cut -d" " -f1`

    if [ $(id -u) -ne 0 ]; then
        local _home_path=`eval echo "~"`
        _log_path="${_home_path}${_log_path}"
        if [ ! -d "${_home_path}" ]; then
            print_log $LEVEL_ERROR "ERR_NO:0x0080;ERR_DES: ${_home_path} does not exist."
            exit 1
        fi
    fi

    logFile="${_log_path}/${INSTALL_LOG_FILE}" # install log path
    optLogFile="${_log_path}/${OPERATION_LOG_FILE}" # operate log path

    if [ ! -d "${_log_path}" ]; then
        createFolder "${_log_path}" "${_cur_user}:${_cur_group}" 750
        if [ $? -ne 0 ]; then
            print_log $LEVEL_WARN "create ${_log_path} failed."
        fi
    fi

    if [ ! -f "${logFile}" ]; then
        createFile "${logFile}" "${_cur_user}:${_cur_group}" 640
        if [ $? -ne 0 ]; then
            print_log $LEVEL_WARN "create $logFile failed."
        fi
    fi

    if [ ! -f "${optLogFile}" ]; then
        createFile "${optLogFile}" "${_cur_user}:${_cur_group}" 640
        if [ $? -ne 0 ]; then
            print_log $LEVEL_WARN "create $optLogFile failed."
        fi
    fi
}

# installed version
getVersionInstalled() {
    version2="none"
    if [ -f "$1/version.info" ]; then
        . "$1/version.info"
        version2=${Version}
    fi
    echo $version2
}

logBaseVersion() {
    installed_version=$(getVersionInstalled "$install_dir/$PACKAGE_NAME")
    if [ ! "${installed_version}"x = ""x ]; then
        log_and_print $LEVEL_INFO "base version is ${installed_version}."
        return 0
    fi
    log_and_print $LEVEL_WARN "base version was destroyed or not exist."
}

getInstallPath() {
    local _temp_path=""
    local _real_install_path=""

    docker_root_path=$(getInstallParam "Docker_Root_Path_Param" "${INSTALL_INFO_FILE}")
    if [ ! -z "${docker_root_path}" ]; then
        docker_flag=y
        _temp_path=$(cd "${docker_root_path}" >/dev/null 2>&1 || exit; pwd)
        if [ -z "${_temp_path}" ]; then
            log_and_print ${LEVEL_ERROR} "The docker root path not exist."
            exit 1
        fi
    fi
    input_install_path=$(getInstallParam "Install_Path_Param" "${INSTALL_INFO_FILE}")
    if [ -z "${_temp_path}" ]; then
        _temp_path=$(cd "${input_install_path}" >/dev/null 2>&1 || exit; pwd)
    else
        _temp_path=$(cd "${_temp_path}/${input_install_path}" >/dev/null 2>&1 || exit; pwd)
    fi
    if [ -z "${_temp_path}" ]; then
        log_and_print ${LEVEL_ERROR} "The install path not exist."
        exit 1
    fi
    # get multi version dir
    get_version_dir "pkg_version_dir" "${VERSION_PATH}"
    if [ ! -z "${pkg_version_dir}" ]; then
        _temp_path=$(cd "${_temp_path}/${pkg_version_dir}" >/dev/null 2>&1 || exit; pwd)
        if [ -z "${_temp_path}" ]; then
            log_and_print ${LEVEL_ERROR} "The cann multi version path not exist."
            exit 1
        fi
    fi
    install_dir=${_temp_path}
}

# get install path
checkInstallPath() {
    check_install_path_valid "${install_dir}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "The install_path $install_dir is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
        exit 1
    fi
}

removeInstallPath() {
    isDirEmpty "${install_dir:?}/$PACKAGE_NAME"
    if [ $? -eq 0 ]; then
        # remove toolkit
        rm -rf "${install_dir:?}/$PACKAGE_NAME"
    fi
    isDirEmpty "${install_dir}"
    if [ $? -eq 0 ]; then
        # remove install path
        rm -rf "${install_dir}"
    fi
    # get multi version dir
    if [ ! -z "$pkg_version_dir" ]; then
        local _ppath=$(dirname "$install_dir")
        isDirEmpty "$_ppath"
        if [ $? -eq 0 ]; then
            rm -rf "$_ppath"
        fi
    fi
}

uninstallRun() {
    local _install_type

    checkDirPermission "$install_dir/$PACKAGE_NAME"
    if [ $? -ne 0 ]; then
        return 1
    fi

    chattr -i -R "$install_dir/$PACKAGE_NAME" > /dev/null 2>&1
    if [ ! -f "$UNINSTALL_SHELL" ]; then
        log_and_print $LEVEL_ERROR "ERR_NO:0X0080;ERR_DES: $UNINSTALL_SHELL does not exist."
        return 1
    fi
    local _uninstall_shell_path=$(readlink -f "$UNINSTALL_SHELL")

    _install_type=$(getInstallParam "Install_Type" "${INSTALL_INFO_FILE}")
    if [ x"${_install_type}" = "x" ]; then
        log_and_print $LEVEL_WARN "The key Install_Type does not exist in $INSTALL_INFO_FILE, and use default $INSTALL_TYPE_ALL."
        "$_uninstall_shell_path" --uninstall "$install_dir" $INSTALL_TYPE_ALL $quiet
    else
        "$_uninstall_shell_path" --uninstall "$install_dir" ${_install_type} $quiet
    fi
    if [ $? -eq 0 ]; then
        log_and_print $LEVEL_INFO "Toolkit package uninstalled successfully! Uninstallation takes effect immediately."

        rm -f "$INSTALL_INFO_FILE"
        removeInstallPath
    else
        log_and_print $LEVEL_ERROR "Toolkit package uninstall failed!"
        return 1
    fi
    return 0
}

source "${COMMON_SHELL}"
source "${COMMON_INC}"

initLog
log_and_print $LEVEL_INFO "LogFile: $logFile"
log_and_print $LEVEL_INFO "OperationLogFile: $optLogFile"

quiet=n
install_dir=""
all_parma="$*"
log_and_print $LEVEL_INFO "InputParams: $all_parma"

while true
do
    case "$1" in
    --quiet)
        quiet=y
        shift
        ;;
    *)
        if [ ! "x$1" = "x" ]; then
            log_and_print $LEVEL_ERROR "ERR_NO:0x0004;ERR_DES: Unrecognized parameters: $1. Only support '--quiet'."
            exit 1
        fi
        break
        ;;
    esac
done

getInstallPath
checkInstallPath
logBaseVersion
uninstallRun
opResult=$?
# log operation log
logOperation $LOG_OPERATION_UNINSTALL $opResult $all_parma
exit $opResult
