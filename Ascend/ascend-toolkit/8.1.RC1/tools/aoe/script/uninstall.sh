#!/bin/bash
# The main script of uninstalling aoe.
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

############### 全局变量定义 ###############
PACKAGE_NAME="aoe"   # 安装包名
PACKAGE_PATH="tools"
HOST="127.0.0.1"
RUN_USERNAME="$(id -un)"    # 当前执行用户
RUN_GROUPNAME="$(id -gn)"   # 当前执行用户组
DEFAULT_USERNAME="HwHiAiUser" # 默认用户
DEFAULT_USERGROUP="HwHiAiUser" # 默认用户属组
DEFAULT_ROOT_DIR="/usr/local/Ascend"  # root 用户默认安装路径
DEFAULT_NORMAL_DIR="${HOME}/Ascend"  # 普通用户默认安装路径

SHELL_PATH="$(dirname "$(readlink -f $0)")" # 脚本目录
COMMON_SHELL="${SHELL_PATH}/common.sh"
UNINSTALL_SHELL="${SHELL_PATH}/run_${PACKAGE_NAME}_uninstall.sh" # 卸载脚本路径
install_path_param="$(dirname "${SHELL_PATH}")"

# info 文件路径
SCENE_INFO_FILE="${install_path_param}/scene.info"
VERSION_INFO_FILE="${install_path_param}/version.info"
ASCEND_INSTALL_INFO_FILE="${install_path_param}/ascend_install.info"
ASCEND_INSTALL_INFO_OLD_FILE="/etc/ascend_install.info"

RUN_CMD="uninstall" # 执行程序命令
RUN_CMD_TYPE="${RUN_CMD^}" # 执行程序命令类型

IS_QUIET="n" # 静默模式默认为 否

# load shell
source "${COMMON_SHELL}"

# 执行程序等级
case "${RUN_CMD_TYPE}" in
    Install)
        LEVEL="SUGGESTION"
        ;;
    Upgrade)
        LEVEL="MINOR"
        ;;
    Uninstall)
        LEVEL="MAJOR"
        ;;
    *)
        LEVEL="UNKNOWN"
        ;;
esac

if [ "$1" ]; then
    RUN_FILE_NAME="$(expr substr $1 5 $(expr ${#1} - 4))" # run file文件名
fi

# 判断当前用户身份,指定日志和安装路径
if [[ $(id -u) -ne 0 ]]; then
    LOG_DIR="${HOME}/var/log/ascend_seclog"  # 普通用户日志存放目录
    DEFAULT_INSTALL_PATH="${HOME}/Ascend"  # 普通用户默认安装路径
else
    LOG_DIR="/var/log/ascend_seclog"  # root 用户日志存放目录
    DEFAULT_INSTALL_PATH="/usr/local/Ascend"  # root 用户默认安装路径
fi

LOG_FILE="${LOG_DIR}/ascend_install.log" # 安装日志文件路径
OPERATION_LOG_FILE="${LOG_DIR}/operation.log"  # operation 日志路径


############### 日志函数 ###############
# 过程日志打印
# 写日志
log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_=${1}
    local msg_="${2}"
    if [ "$log_type_" == "INFO" ]; then
        local log_format_="[Aoe] [$cur_date_] [$log_type_]: ${msg_}"
        echo "$log_format_"
    elif [ "$log_type_" == "WARNING" ]; then
        local log_format_="[Aoe] [$cur_date_] [$log_type_]: ${msg_}"
        echo "$log_format_"
    elif [ "$log_type_" == "ERROR" ]; then
        local log_format_="[Aoe] [$cur_date_] [$log_type_]: ${msg_}"
        echo "$log_format_"
    elif [ "$log_type_" == "DEBUG" ]; then
        local log_format_="[Aoe] [$cur_date_] [$log_type_]: ${msg_}"
    fi
    echo "$log_format_" >> $LOG_FILE
}

# 静默模式日志打印
new_echo() {
    local log_type_=${1}
    local log_msg_="${2}"
    if  [ "${is_quiet}" = "n" ]; then
        log ${log_type_} ${log_msg_} 1 > /dev/null
    fi
}

# 开始执行前打印开始信息
start_log() {
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    log "INFO" "Start time:${cur_date}"
    log "INFO" "LogFile:${LOG_FILE}"
    log "INFO" "InputParams:--${RUN_CMD}"
}

# 退出时打印结束日志
exit_log() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    new_echo "INFO" "End time:${cur_date}"
    log "INFO" "End time:${cur_date}"
    exit $1
}

# 打印 Operation 日志
log_operation() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    if [ ! -f "${OPERATION_LOG_FILE}" ]; then
        touch ${OPERATION_LOG_FILE}
        chmod 640 ${OPERATION_LOG_FILE}
    fi
    echo "${RUN_CMD_TYPE} ${LEVEL} ${RUN_USERNAME} ${cur_date} ${HOST} ${RUN_FILE_NAME} $2 installmode=${RUN_CMD}; cmdlist=--${RUN_CMD}" >> ${OPERATION_LOG_FILE}
}


############### 错误函数 ###############ls
# 不支持的参数
err_no0x0004() {
    log "ERROR" "ERR_NO:0x0004;ERR_DES: Unrecognized parameters: $1"
    exit_log 1
}

# 文件没有找到
err_no0x0080() {
    log "ERROR" "ERR_NO:0x0080;ERR_DES:This file or directory does not exist,$1"
    exit_log 1
}

# 用户权限不足
err_no0x0093() {
    log "ERROR" "ERR_NO:0x0093;ERR_DES:Permission denied,$1"
    exit_log 1
}


############### 环境适配函数 ###############
chmod_start() {
    chmod -Rf 750 "${install_path_param}"
}
############### 检验函数 ###############
# 用户权限认证
user_auth(){
    local dir_user_id=$(stat -c "%u" "${install_path_param}")
    local run_user_id=$(id -u)
    if [[ ${run_user_id} -ne 0 ]]; then
        if [[ ${run_user_id} -ne ${dir_user_id} ]]; then
            err_no0x0093 "Current user is not supported to ${RUN_CMD} the ${PACKAGE_NAME} package"
        fi
    fi
}

# 目录是否为空
is_dir_empty(){
    local dir_file_num=$(ls "$1" | wc -l)
    if [[ ${dir_file_num} -eq 0 ]]; then
        return 0    # 空目录返回0
    else
        return 1    # 非空目录返回1
    fi
}

save_user_files_to_log(){
    if [ "$1" = "${install_path_param}" ] && [ -s "$1" ]; then
        local file_num=$(ls -lR "$1"|grep "^-"|wc -l)
        local dir_num=$(ls -lR "$1"|grep "^d"|wc -l)
        local total_num=$(expr ${file_num} + ${dir_num})
        if [ $total_num -eq 2 ]; then
            if [ -f "${VERSION_INFO_FILE}" ] && [ -f "${ASCEND_INSTALL_INFO_FILE}" ]; then
                return 0
            fi
        fi
        if [ $total_num -eq 1 ]; then
            if [ -f "${VERSION_INFO_FILE}" ] || [ -f "${ASCEND_INSTALL_INFO_FILE}" ]; then
                return 0
            fi
        fi
        log "INFO" "Some files generated by user are not cleared, if necessary, manually clear them, get details in $LOG_FILE"
    fi
    if [ -s "$1" ]; then
        for file in $(ls -a "$1"); do
            if test -d "$1/$file"; then
                if [[ "$file" != '.' && "$file" != '..' ]]; then
                    echo "$1/$file" >> $LOG_FILE
                    save_user_files_to_log "$1/$file"
                fi
            else
                echo "$1/$file" >> $LOG_FILE
            fi
        done
    fi
}

############### 执行函数 ###############
uninstall_run() {
    user_auth
    chmod_start
    local operation="${RUN_CMD_TYPE}"
    if [ -f "${ASCEND_INSTALL_INFO_FILE}" ]; then
        aoe_install_path_param=$(get_install_param "Aoe_Install_Path_Param" "${ASCEND_INSTALL_INFO_FILE}")
        aoe_install_type=$(get_install_param "Aoe_Install_Type" "${ASCEND_INSTALL_INFO_FILE}")
    elif [ -f "${ASCEND_INSTALL_INFO_OLD_FILE}" ]; then
        num=$(grep -c -i "Aoe_Install_Path_Param" ${ASCEND_INSTALL_INFO_OLD_FILE})
        if [[ ${num} -ne 0 ]]; then
            . ${ASCEND_INSTALL_INFO_OLD_FILE}
        fi
    else
        err_no0x0080 "please complete ${ASCEND_INSTALL_INFO_FILE} or ${ASCEND_INSTALL_INFO_OLD_FILE}"
    fi
    if [[ $? -eq 0 ]]; then
        log "INFO" "${RUN_CMD} ${aoe_install_path_param} ${aoe_install_type}"
        bash "${UNINSTALL_SHELL}" ${RUN_CMD} "${aoe_install_path_param}" ${aoe_install_type} ${IS_QUIET}
        if [[ $? -eq 0 ]]; then
            rm -f "${ASCEND_INSTALL_INFO_FILE}"
            if [[ $? -eq 0 ]] && [ -f "${ASCEND_INSTALL_INFO_OLD_FILE}" ] && [[ ${num} -ne 0 ]]; then
                sed -i '/Aoe_Install_Path_Param=/d' ${ASCEND_INSTALL_INFO_OLD_FILE}
                sed -i '/Aoe_Install_Path_Param=/d' ${ASCEND_INSTALL_INFO_OLD_FILE}
            fi
            remove_dir_recursive ${aoe_install_path_param} ${install_path_param}
            if [ "$(ls -A "$aoe_install_path_param")" = "" ]; then
                test -d "$aoe_install_path_param" && rm -rf "$aoe_install_path_param"
            fi
            new_echo "INFO" "Aoe package uninstalled successfully! Uninstallation takes effect immediately."
            log "INFO" "Aoe package uninstalled successfully! Uninstallation takes effect immediately."
            log_operation "${operation}" "success"
            save_user_files_to_log "${aoe_install_path_param}/${pkg_relative_path}"
        else
            log "WARNING" "${operation} failed"
            log_operation "${operation}" "failed"
            exit_log 1
        fi
    fi
    return $?
}
############### 程序执行 ###############
while true
do
    case "$1" in
    --quiet)
        IS_QUIET="y"
        shift
        ;;
    *)
        if [ ! "$1" = "" ]; then
            err_no0x0004 "$1 . Only support '--quiet'."
        fi
        break
        ;;
    esac
done

# 输出执行开始日志
start_log

# 验证此目录是否为空
is_dir_empty "${install_path_param}"

if [[ $? -ne 0 ]]; then
    # 执行卸载
    uninstall_run
else
    # 报错
    err_no0x0080 "runfile is not installed on this device, uninstall failed"
fi

exit_log 0

