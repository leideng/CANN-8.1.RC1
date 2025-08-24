#!/bin/sh
# Perform install/upgrade/uninstall for packages
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

CURPATH=$(dirname $(readlink -f "$0"))
DEFAULT_ROOT_DIR="/usr/local/Ascend"
DEFAULT_NORMAL_DIR="${HOME}/Ascend"
USERNAME=$(id -un)
USERGROUP=$(id -gn)
common_func_path="${CURPATH}/common_func.inc"
version_compat_func_path="${CURPATH}/version_compatiable.inc"
common_func_v2_path="${CURPATH}/common_func_v2.inc"
version_cfg_path="${CURPATH}/version_cfg.inc"
pkg_version_path="${CURPATH}/../../version.info"

. "${common_func_path}"
. "${version_compat_func_path}"
. "${common_func_v2_path}"
. "${version_cfg_path}"

unique_mode() {
    if [ ! -z "$g_param_check_flag" ]; then
        comm_log "ERROR" "Only support one type: full, operation failed!"
        exit 1
    fi
    g_param_check_flag="true"
}

output_progress() {
    comm_log "INFO" "install upgradePercentage:$1%"
}

do_install_run() {
    local package="$1"
    local install_path="$2"
    local install_type="$3"
    local feature_type="$4"
    local version_info="$5"

    # 执行安装
    sh "$CURPATH/install_common_parser.sh" --package="$package" --install --username="$USERNAME" --usergroup="$USERGROUP" \
        --version-file="$version_info" --set-cann-uninstall "$install_type" "$install_path" "$CURPATH/filelist.csv" "$feature_type"
    if [ $? -ne 0 ]; then
        return 1
    fi

    return 0
}

install_run() {
    comm_create_package_dir "$PACKAGE_DIR" "$USERNAME:$USERGROUP" "$INSTALL_FOR_ALL"
    comm_create_install_info_by_path "$INSTALL_INFO_PATH" "$USERNAME:$USERGROUP"
    comm_update_install_info --install-path-param="$install_path_param" --install-type="$INSTALL_TYPE" \
                             --username="$USERNAME" --usergroup="$USERGROUP" --feature-type="$FEATURE_TYPE" \
                             "$PACKAGE" "$INSTALL_INFO_PATH"

    comm_log "INFO" "install target dir $install_path_param, type $INSTALL_TYPE."
    do_install_run "$PACKAGE" "$install_path_param" "$INSTALL_TYPE" "$FEATURE_TYPE" "$pkg_version_path"
}

OPERATION=""
INSTALL_TYPE=""
FEATURE_TYPE="all"
IS_QUIET=n
PACKAGE=""
g_param_check_flag=""

if [ $(id -u) -eq 0 ]; then
    INSTALL_FOR_ALL=y
    install_path_param="$DEFAULT_ROOT_DIR"
else
    INSTALL_FOR_ALL=n
    install_path_param="$DEFAULT_NORMAL_DIR"
fi

while true
do
    case "$1" in
    --package=*)
        PACKAGE="$(echo "$1" | cut -d"=" -f2-)"
        shift;
        ;;
    *)
        break
        ;;
    esac
done

####################################################################################################

RUNFILENAME=$(expr substr "$1" 5 $(expr ${#1} - 4))

get_titled_package_name "TITLED_PACKAGE" "$PACKAGE"

# 设置日志参数
set_comm_log "$TITLED_PACKAGE" "$COMM_LOGFILE"
comm_init_log

####################################################################################################
if [ "$#" = "1" ] || [ "$#" = "2" ]; then
    comm_log "ERROR" "Unrecognized parameters. Try './xxx.run --help for more information.'"
    exit 1
fi

i=0
while true
do
    if [ "$1" = "--" ]; then
        break
    fi
    if [ "$(expr substr "$1" 1 2)" = "--" ]; then
        i=$(expr $i + 1)
    fi
    if [ $i -gt 2 ]; then
        break
    fi
    shift 1
done

ALL_PARAM="$@"

#################################################################################
while true
do
    case "$1" in
    --help | -h)
        comm_print_usage "$RUNFILENAME"
        exit 0
        ;;
    --run)
        unique_mode
        OPERATION="install"
        INSTALL_TYPE="run"
        shift
        ;;
    --devel)
        unique_mode
        OPERATION="install"
        INSTALL_TYPE="devel"
        shift
        ;;
    --full)
        unique_mode
        OPERATION="install"
        INSTALL_TYPE="full"
        shift
        ;;
    --upgrade)
        unique_mode
        OPERATION="upgrade"
        shift
        ;;
    --uninstall)
        unique_mode
        OPERATION="uninstall"
        shift
        ;;
    --feature=*)
        FEATURE_TYPE=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --install-path=*)
        temp_path="$(echo "$1" | cut -d"=" -f2-)"
        comm_parse_install_path "install_path_param" "$temp_path" "$PACKAGE"
        shift
        ;;
    --quiet)
        IS_QUIET=y
        shift
        ;;
    --extract=*)
        shift;
        ;;
    -*)
        comm_log "ERROR" "Unsupported parameters : $1"
        comm_print_usage "$RUNFILENAME"
        exit 0
        ;;
    *)
        break
        ;;
    esac
done

get_install_package_dir "PACKAGE_DIR" "$pkg_version_path" "$install_path_param" "$PACKAGE"
INSTALL_INFO_PATH="$PACKAGE_DIR/ascend_install.info"

comm_start_log "$ALL_PARAM"

uninstall_run() {
    local skip_recreate_softlink="$1"
    local skip_end_log="$2"
    local uninstall_option=""
    if [ ! -d "$PACKAGE_DIR" ]; then
        comm_log "ERROR" "$TITLED_PACKAGE package isn't installed, uninstall failed!"
        comm_log_operation "Uninstall" "$RUNFILENAME" "failed" "" "$ALL_PARAM"
        comm_exit_log 1
    fi
    if [ ! -f "$PACKAGE_DIR/script/common_uninstall.sh" ]; then
        comm_log "ERROR" "Can't find $TITLED_PACKAGE package uninstall script, uninstall failed!"
        comm_log_operation "Uninstall" "$RUNFILENAME" "failed" "" "$ALL_PARAM"
        comm_exit_log 1
    fi
    if [ "$skip_recreate_softlink" = "y" ]; then
        uninstall_option="$uninstall_option --skip-recreate-softlink"
    fi
    if [ "$skip_end_log" = "y" ]; then
        uninstall_option="$uninstall_option --skip-end-log"
    fi
    sh "$PACKAGE_DIR/script/common_uninstall.sh" --package="$PACKAGE" --runfilename="$RUNFILENAME" --skip-start-log $uninstall_option -- $ALL_PARAM
}


if [ "$OPERATION" = "install" ]; then
    install_run
    if [ $? -ne 0 ]; then
        comm_log "ERROR" "$TITLED_PACKAGE package install failed, please retry after uninstall!"
        comm_log_operation "Install" "$RUNFILENAME" "failed" "$INSTALL_TYPE" "$ALL_PARAM"
        comm_exit_log 1
    else
        comm_log "INFO" "$TITLED_PACKAGE package installed successfully! The new version takes effect immediately."
        comm_log_operation "Install" "$RUNFILENAME" "succeeded" "$INSTALL_TYPE" "$ALL_PARAM"
        comm_exit_log 0
    fi
elif [ "$OPERATION" = "upgrade" ]; then
    if [ ! -f "$INSTALL_INFO_PATH" ]; then
        comm_err_file_or_directory_not_exist "$INSTALL_INFO_PATH"
    fi
    comm_get_install_param INSTALL_TYPE "$INSTALL_INFO_PATH" "${PACKAGE}_Install_Type"
    comm_get_install_param FEATURE_TYPE "$INSTALL_INFO_PATH" "${PACKAGE}_Feature_Type"

    uninstall_run "y" "y"
    ret="$?" && [ $ret -ne 0 ] && exit $ret

    install_run
    if [ $? -ne 0 ]; then
        comm_log "ERROR" "$TITLED_PACKAGE package upgrade failed, please retry after uninstall!"
        comm_log_operation "Upgrade" "$RUNFILENAME" "failed" "$INSTALL_TYPE" "$ALL_PARAM"
        comm_exit_log 1
    else
        comm_log "INFO" "$TITLED_PACKAGE package upgraded successfully! The new version takes effect immediately."
        comm_log_operation "Upgrade" "$RUNFILENAME" "succeeded" "$INSTALL_TYPE" "$ALL_PARAM"
        comm_exit_log 0
    fi
elif [ "$OPERATION" = "uninstall" ]; then
    uninstall_run "n" "n"
    ret="$?" && [ $ret -ne 0 ] && exit $ret
fi
