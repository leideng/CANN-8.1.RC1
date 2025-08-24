#!/bin/bash

# 请在此处定义各种变量
readonly PACKAGE_SHORT_NAME="toolkit"
readonly PACKAGE_VERSION_FORM="aarch64-linux"
readonly PACKAGE_LOG_NAME="Toolkit"
readonly PACKAGE_ARCH="aarch64"
readonly PYTHON3_INSTALL_CONF="use_private_python.info"
readonly PATH_LENGTH=4096

frame=$(arch)
install_for_all_cmd=""
log_file_name="ascend_${PACKAGE_SHORT_NAME}_install.log" # log文件名字
log_file=""                                              # log文件带路径
if [ "$UID" = "0" ]; then
    LOG_PATH="/var/log/ascend_seclog"
    log_file="${LOG_PATH}/${log_file_name}"
    install_for_all_cmd="--install-for-all"
    PYTHON3_INSTALL_INFO="/etc/${PYTHON3_INSTALL_CONF}"
else
    LOG_PATH="${HOME}/var/log/ascend_seclog"
    log_file="${LOG_PATH}/${log_file_name}"
    PYTHON3_INSTALL_INFO="${HOME}/${PYTHON3_INSTALL_CONF}"
fi

# 路径
script_path="$(dirname $(readlink -f $0))"
install_path="$(cd "$(dirname ${script_path})" && pwd)"
form_path=$([ x${PACKAGE_VERSION_FORM} == x"" ] && echo ${install_path} || echo $(dirname ${install_path}))

# 日志模块初始化
function log_init() {
    # 判断输入的安装路径路径是否存在，不存在则创建
    if [ ! -d $LOG_PATH ]; then
        make_dir "$LOG_PATH"
    fi
    if [ ! -f $log_file ]; then
        make_file "$log_file"
        chmod_recursion ${log_file} "640" "log"
        log "INFO" "Log file not found, create a new log file."
    else
        local filesize=$(ls -l $log_file | awk '{ print $5 }')
        local maxsize=$((1024 * 1024 * 50))
        if [ $filesize -gt $maxsize ]; then
            local log_file_move_name="ascend_${PACKAGE_SHORT_NAME}_install_bak.log"
            mv -f ${log_file} ${LOG_PATH}/${log_file_move_name}
            chmod_recursion ${LOG_PATH}/${log_file_move_name} "440" "log"
            make_file "$log_file"
            chmod_recursion ${log_file} "640" "log"
            log "INFO" "log file > 50M, move ${log_file} to ${LOG_PATH}/${log_file_move_name}."
        fi
    fi
    print "INFO" "LogFile:$log_file"
}

# 权限掩码设置
function change_umask() {
    if [ ${UID} -eq 0 ] && [ $(umask) != "0022" ]; then
        print "INFO" "change umask 0022"
        umask 0022
    elif [ ${UID} -ne 0 ] && [ $(umask) != "0002" ]; then
        print "INFO" "change umask 0002"
        umask 0002
    fi
}

# 创建文件夹
function make_dir() {
    change_umask
    print "INFO" "mkdir ${1}"
    mkdir -p ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 fail !"
        exit 1
    fi
}

# 创建文件
function make_file() {
    change_umask
    print "INFO" "touch ${1}"
    touch ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 fail !"
        exit 1
    fi
}

# 递归授权
function chmod_recursion() {
    local parameter2=$2
    local rights="$(echo ${parameter2:0:2})""$(echo ${parameter2:1:1})"
    rights=$([ x${install_for_all_cmd} == x"" ] && echo $2 || echo ${rights})
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -exec chmod ${rights} {} \; 2>/dev/null
        # 日志文件不增加other权限
    elif [ "$3" = "log" ]; then
        find $1 -type f -exec chmod ${parameter2} {} \; 2>/dev/null
    fi
}

# 将日志打印到文件中
function log() {
    if [ x$log_file = x ] || [ ! -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
    elif [ -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2" >>$log_file
    fi
}

# 将关键信息打印到屏幕上
function print() {
    if [ x$log_file = x ] || [ ! -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
    else
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2" | tee -a $log_file
    fi
}

# 检查路径字符串
function check_path() {
    local path_str=${1}
    # 判断路径字符串长度
    if [ ${#path_str} -gt ${PATH_LENGTH} ]; then
        print "WARNING" "parameter error $path_str, the length exceeds ${PATH_LENGTH}."
        return 1
    fi
    # 判断是否是绝对路径
    if [[ ! "${path_str}" =~ ^/.* ]]; then
        print "WARNING" "parameter error $path_str, must be an absolute path."
        return 1
    fi
    # 黑名单设置，不允许//，...这样的路径
    if echo "${path_str}" | grep -Eq '\/{2,}|\.{3,}'; then
        print "WARNING" "The path ${path_str} is invalid, cannot contain the following characters: // ...!"
        return 1
    fi
    # 白名单设置，只允许常见字符
    if echo "${path_str}" | grep -Eq '^\~?[a-zA-Z0-9./_-]*$'; then
        log "INFO" "The path ${path_str} is correct."
        return 0
    else
        print "WARNING" "The path ${path_str} is invalid, only [a-z,A-Z,0-9,-,_] is support!"
        return 1
    fi
}

# python3环境变量导入
function set_python_environment() {
    if [ -f ${PYTHON3_INSTALL_INFO} ]; then
        local param1=$(cat "${PYTHON3_INSTALL_INFO}" | grep -w "python37_install_path" | cut -d"=" -f2 | sed "s/ //g")
        local param2=$(cat "${PYTHON3_INSTALL_INFO}" | grep -w "python3_install_path" | cut -d"=" -f2 | sed "s/ //g")
        local python3_install_path=$([ x${param1} == x"" ] && echo ${param2} || echo ${param1})
        if [ x"${python3_install_path}" == x"" ]; then
            print "WARNING" "the ${PYTHON3_INSTALL_INFO} file has no python37_install_path or python3_install_path variable. Please check and set it."
        else
            check_path "${python3_install_path}"
            if [ $? -ne 0 ]; then
                print "WARNING" "the ${PYTHON3_INSTALL_INFO} file Python path ${python3_install_path} is error."
            elif [ ! -d "${python3_install_path}" ]; then
                print "WARNING" "the ${PYTHON3_INSTALL_INFO} file Python path ${python3_install_path} is not exist or not directory, please check it."
            else
                if ls ${python3_install_path}/bin/python* 1>/dev/null 2>&1; then
                    export LD_LIBRARY_PATH=${python3_install_path}/lib/:$LD_LIBRARY_PATH
                    export PATH=${python3_install_path}/bin/:$PATH
                    log "INFO" "Setting environment variables ${python3_install_path} succeeded."
                else
                    print "WARNING" "The ${PYTHON3_INSTALL_INFO} file environment variable ${python3_install_path}/bin has no Python binary file, please check and reset it."
                fi
            fi
        fi
    fi
}

# 程序开始
function main() {
    # 日志初始化,后续所有模块都有可能使用日志模块必须最先初始化
    log_init
    # python3 环境变量导入
    set_python_environment
    # 图灵总卸载脚本调用
    if [ x"${PACKAGE_VERSION_FORM}" == x"" ] || [ ${frame} == ${PACKAGE_ARCH} ]; then
        if [ -f "${form_path}/cann_uninstall.sh" ]; then
            ${form_path}/cann_uninstall.sh | tee -a $log_file
        else
            print "ERROR" "cann_uninstall.sh file not found"
            exit 1
        fi
    else
        if [ -f "${install_path}/hetero-arch-scripts/cann_uninstall.sh" ]; then
            ${install_path}/hetero-arch-scripts/cann_uninstall.sh | tee -a $log_file
        elif [ -f "${form_path}/cann_uninstall.sh" ]; then
            ${form_path}/cann_uninstall.sh | tee -a $log_file
        else
            print "ERROR" "cann_uninstall.sh file not found"
            exit 1
        fi
    fi
    if [ $? -eq 0 ]; then
        print "INFO" "${PACKAGE_SHORT_NAME} uninstall success"
        exit 0
    else
        print "ERROR" "${PACKAGE_SHORT_NAME} uninstall failed, Please refer to the log for more details: ${log_file}"
        exit 1
    fi
}

main
