#!/bin/sh
#----------------------------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#----------------------------------------------------------------------------

# 通知latest管理器创建版本软链
notify_latest_manager_create_version_softlink() {
    local curpath install_path version_dir var_path

    set_comm_log "Notifier"

    curpath="$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))"
    install_path="$(readlink -f "$curpath/..")"
    version_dir="$(basename "$curpath")"
    var_path="$install_path/$LATEST_DIR/var"

    if [ ! -f "$var_path/manager.sh" ]; then
        comm_log "ERROR" "$var_path/manager.sh doesn't exist!"
        exit 2
    fi

    if ! "$var_path/manager.sh" --version-dir "$version_dir" create_version_softlink; then
        comm_log "ERROR" "create version softlink failed!"
        exit 1
    fi
    return 0
}

# 通知latest管理器删除latest软链
notify_latest_manager_remove_latest_softlink() {
    local curpath install_path var_path

    set_comm_log "Notifier"

    curpath="$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))"
    install_path="$(readlink -f "$curpath/..")"
    var_path="$install_path/$LATEST_DIR/var"

    if [ ! -f "$var_path/manager.sh" ]; then
        comm_log "ERROR" "$var_path/manager.sh doesn't exist!"
        exit 2
    fi

    if ! "$var_path/manager.sh" remove_latest_softlink; then
        comm_log "ERROR" "remove latest softlink failed!"
        exit 1
    fi
    return 0
}

## module log

if [ "$(id -u)" != "0" ]; then
    COMM_LOG_DIR="${HOME}/var/log/ascend_seclog"
else
    COMM_LOG_DIR="/var/log/ascend_seclog"
fi

COMM_OPERATION_LOGFILE="${COMM_LOG_DIR}/operation.log"
COMM_LOGFILE="${COMM_LOG_DIR}/ascend_install.log"
COMM_USERNAME="$(id -un)"
COMM_USERGROUP="$(id -gn)"

LOG_PKG_NAME="Common"
LOG_FILE="/dev/null"
LOG_STYLE="normal"

# 初始化日志系统
comm_init_log() {
    if [ ! -d "$COMM_LOG_DIR" ]; then
        mkdir -p "$COMM_LOG_DIR"
    fi
    if [ $(id -u) -ne 0 ]; then
        chmod 740 "$COMM_LOG_DIR"
    else
        chmod 750 "$COMM_LOG_DIR"
    fi
    if [ ! -f "$COMM_LOGFILE" ]; then
        touch "$COMM_LOGFILE"
    fi
    chmod 640 "$COMM_LOGFILE"
}

# 组合格式化后的日志信息
# _outvar : [输出变量] 格式化后的日志信息
# _log_type : 日志级别
# _msg : 日志信息
_comm_compose_log_msg() {
    local _outvar="$1"
    local _log_type="$2"
    local _msg="$3"
    local _cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local _result

    if [ "${LOG_STYLE}" = "no-colon" ]; then
        _result="[${LOG_PKG_NAME}] [${_cur_date}] [${_log_type}]${_msg}"
    else
        _result="[${LOG_PKG_NAME}] [${_cur_date}] [${_log_type}]: ${_msg}"
    fi
    eval "${_outvar}=\"${_result}\""
}

# 输出格式化后的消息
_comm_echo_log_msg() {
    local log_type_="$1"
    local log_format_="$2"

    if [ $log_type_ = "INFO" ]; then
        echo "${log_format_}"
    elif [ $log_type_ = "WARNING" ]; then
        echo "${log_format_}"
    elif [ $log_type_ = "ERROR" ]; then
        echo "${log_format_}"
    elif [ $log_type_ = "DEBUG" ]; then
        :
    fi
}

# 输出
comm_echo() {
    local log_type="$1"
    local msg="$2"
    local log_msg

    _comm_compose_log_msg "log_msg" "$log_type" "$msg"
    _comm_echo_log_msg "$log_type" "$log_msg"
}

# 写日志
comm_log() {
    local log_type="$1"
    local msg="$2"
    local log_msg

    _comm_compose_log_msg "log_msg" "$log_type" "$msg"
    _comm_echo_log_msg "$log_type" "$log_msg"
    echo "$log_msg" >> "${LOG_FILE}"
}

# 设置日志参数
set_comm_log() {
    local pkg_name="$1"
    local log_file="$2"

    LOG_PKG_NAME="${pkg_name}"
    if [ "$log_file" != "" ]; then
        LOG_FILE="${log_file}"
    fi
}

# 安全日志
comm_log_operation() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local operation="$1"
    local runfilename="$2"
    local result="$3"
    local installmode="$4"
    local all_parma="$5"
    local level=""
    if [ "${operation}"  = "Install" ]; then
        level="SUGGESTION"
    elif [ "${operation}" = "Upgrade" ]; then
        level="MINOR"
    elif  [ "${operation}" = "Uninstall" ]; then
        level="MAJOR"
    else
        level="UNKNOWN"
    fi

    if [ ! -f "${COMM_OPERATION_LOGFILE}" ]; then
        touch "${COMM_OPERATION_LOGFILE}"
        chmod 640 "${COMM_OPERATION_LOGFILE}"
    fi

    echo "${operation} ${level} ${COMM_USERNAME} ${cur_date} 127.0.0.1 ${runfilename} ${result} installmode=${installmode}; cmdlist=${all_parma}" >> "${COMM_OPERATION_LOGFILE}"
}

## end module

LATEST_DIR="latest"

notify_latest_manager_remove_latest_softlink
