#!/bin/bash
# Perform uninstall for packages
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

############### 全局变量定义 ###############

CURPATH="$(dirname $(readlink -f $0))" # 脚本目录
UNINSTALL_SHELL="${CURPATH}/run_compiler_uninstall.sh" # 卸载脚本路径
INSTALL_COMMON_PARSER="${CURPATH}/install_common_parser.sh"
FILELIST_PATH="${CURPATH}/filelist.csv"

PACKAGE_DIR="$(dirname "$CURPATH")"
VERSION_INFO_PATH="$PACKAGE_DIR/version.info"
INSTALL_INFO_PATH="$PACKAGE_DIR/ascend_install.info"

COMMON_FUNC_PATH="${CURPATH}/common_func.inc"
COMMON_FUNC_V2_PATH="${CURPATH}/common_func_v2.inc"

. "$COMMON_FUNC_PATH"
. "$COMMON_FUNC_V2_PATH"

PACKAGE=""
RUNFILENAME=""             # run包名
IS_QUIET="n"               # 静默模式默认为否
IS_RECREATE_SOFTLINK="y"   # 是否重建latest软链
SKIP_START_LOG="n"         # 跳过开始日志
SKIP_END_LOG="n"           # 跳过结束日志

INSTALL_TYPE=""            # 安装类型

############### 程序执行 ###############
while true
do
    case "$1" in
    --package=*)
        PACKAGE="$(echo "$1" | cut -d"=" -f2-)"
        shift;
        ;;
    --runfilename=*)
        RUNFILENAME="$(echo "$1" | cut -d"=" -f2-)"
        shift;
        ;;
    --quiet)
        IS_QUIET="y"
        shift
        ;;
    --skip-recreate-softlink)
        IS_RECREATE_SOFTLINK="n"
        shift
        ;;
    --skip-start-log)
        SKIP_START_LOG="y"
        shift
        ;;
    --skip-end-log)
        SKIP_END_LOG="y"
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        break
        ;;
    esac
done

if [ "$PACKAGE" = "" ]; then
    comm_echo "ERROR" "the following arguments are required: --package"
    exit 1
fi

ALL_PARAM="$@"
if [ "$ALL_PARAM" = "" ]; then
    ALL_PARAM="--uninstall"
fi

if [ "$RUNFILENAME" = "" ]; then
    RUNFILENAME="$PACKAGE"
fi

get_titled_package_name "TITLED_PACKAGE" "$PACKAGE"

set_comm_log "$TITLED_PACKAGE" "$COMM_LOGFILE"
comm_init_log

if [ "$SKIP_START_LOG" = "n" ]; then
    # 输出执行开始日志
    comm_start_log "$ALL_PARAM"
fi

############### 检验函数 ###############
# 用户权限认证
user_auth() {
    local dir_user_id=$(stat -c "%u" "$PACKAGE_DIR")
    local run_user_id=$(id -u)
    if [ "${run_user_id}" -ne 0 ]; then
        if [ "${run_user_id}" -ne "${dir_user_id}" ]; then
            comm_log "ERROR" "Current user is not supported to uninstall the $PACKAGE package"
        fi
    fi
}

do_uninstall_run() {
    if [ "${IS_RECREATE_SOFTLINK}" = "y" ]; then
        recreate_softlink_option="--recreate-softlink"
    else
        recreate_softlink_option=""
    fi

    # 执行卸载
    sh "$INSTALL_COMMON_PARSER" --package="$PACKAGE" --uninstall --username="$COMM_USERNAME" --usergroup="$COMM_USERGROUP" \
       $recreate_softlink_option --version-file="$VERSION_INFO_PATH" --remove-install-info \
       "$INSTALL_TYPE" "$INSTALL_PATH_PARAM" "$FILELIST_PATH"
    if [ $? -ne 0 ]; then
        return 1
    fi

    if [ -n "$latest_path" ] && [ -d "$latest_path" ] && [ "x$(ls -A $latest_path 2>&1)" = "x" ]; then
        rm -rf "$latest_path"
    fi

    return 0
}

############### 执行函数 ###############
uninstall_run() {
    user_auth

    comm_log "INFO" "uninstall target dir $INSTALL_PATH_PARAM, type $INSTALL_TYPE."
    do_uninstall_run
    return $?
}

if [ ! -f "$INSTALL_INFO_PATH" ]; then
    comm_err_file_or_directory_not_exist "$INSTALL_INFO_PATH"
fi

if [ ! -f "$VERSION_INFO_PATH" ]; then
    comm_err_file_or_directory_not_exist "$VERSION_INFO_PATH"
fi

comm_get_install_param INSTALL_TYPE "$INSTALL_INFO_PATH" "${PACKAGE}_Install_Type"
comm_get_install_param INSTALL_PATH_PARAM "$INSTALL_INFO_PATH" "${PACKAGE}_Install_Path_Param"

# 执行卸载
uninstall_run
if [ $? -eq 0 ]; then
    remove_dir_if_empty "$INSTALL_PATH_PARAM"
    comm_log "INFO" "$PACKAGE package uninstalled successfully! Uninstallation takes effect immediately."
    comm_log_operation "Uninstall" "$PACKAGE" "succeeded" "$INSTALL_TYPE" "$ALL_PARAM"
    if [ "$SKIP_END_LOG" = "n" ]; then
        comm_exit_log 0
    else
        exit 0
    fi
else
    comm_log "ERROR" "failed to uninstall package."
    comm_log_operation "Uninstall" "$PACKAGE" "failed" "$INSTALL_TYPE" "$ALL_PARAM"
    if [ "$SKIP_END_LOG" = "n" ]; then
        comm_exit_log 1
    else
        exit 1
    fi
fi
