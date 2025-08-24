#!/bin/bash
INSTALL_TYPE_ALL=full # run/devel/docker
INSTALL_TYPE_RUN=run
INSTALL_TYPE_DEV=devel
DEFAULT_USERNAME=${USER}
DEFAULT_USERGROUP=$(groups | cut -d" " -f1)
PACKAGE_NAME=toolkit
LEVEL_INFO="INFO"
LEVEL_WARN="WARNING"
LEVEL_ERROR="ERROR"

INSTALL_INFO_FILE=ascend_install.info
INSTALL_LOG_FILE=ascend_install.log
SHELL_DIR=$(cd "$(dirname "$0")" || exit;pwd)
INSTALL_COMMON_PARSER_PATH="$SHELL_DIR/install_common_parser.sh"
FILELIST_PATH="$SHELL_DIR/filelist.csv"
COMMON_SHELL_PATH="$SHELL_DIR/common.sh"
COMMON_INC="$SHELL_DIR/common_func.inc"
VERSION_INFO="$SHELL_DIR/../version.info"
LOG_RELATIVE_PATH=var/log/ascend_seclog # install log path and operation log path

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

if [ "$(id -u)" -ne 0 ]; then
    home_path=`eval echo "~${USER}"`
    logFile="$home_path/$LOG_RELATIVE_PATH/$INSTALL_LOG_FILE"
else
    logFile="/$LOG_RELATIVE_PATH/$INSTALL_LOG_FILE"
fi

##########################################################################
log $LEVEL_INFO "step into run_${PACKAGE_NAME}_uninstall.sh ..."

if [ $# -ne 4 ]; then
    log_and_print $LEVEL_ERROR "input params number error."
    exit 1
fi
install_dir="$2"
install_type="$3"
quiet="$4"

removeVersionInfo() {
    # remove version.info
    local _version_info="${install_dir}/${PACKAGE_NAME}/version.info"

    if [ -f "${_version_info}" ]; then
        rm -f "${_version_info}"
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_WARN "ERR_NO:0x0090;ERR_DES: Remove version.info failed."
        fi
    fi
}

uninstallTool()
{
    if [ ! -f "$VERSION_INFO" ]; then
        log_and_print $LEVEL_ERROR "Version info file not exist."
        return 1
    fi
    local shell_options_="--package=${PACKAGE_NAME} --recreate-softlink --username=${username} \
--usergroup=${usergroup} --docker-root=${docker_root_path}"
    . "${COMMON_INC}"
    is_multi_version_pkg "is_multi_version_" "${VERSION_INFO}"
    if [ "${is_multi_version_}" = "true" ]; then
        get_version "version_" "${VERSION_INFO}"
        get_version_dir "version_dir_" "${VERSION_INFO}"
        shell_options_="${shell_options_} --version=${version_} --version-dir=${version_dir_}"
    fi
    if [ "-${setenv}" = "-y" ]; then
        shell_options_="${shell_options_} --setenv"
    fi
    if [ "-${install_for_all}" = "-y" ]; then
        shell_options_="${shell_options_} --install_for_all"
    fi
    local custom_options_="--custom-options=--logfile=$logFile,--quiet=$quiet,--pylocal=$pylocal,\
--feature=$feature_type --chip=$chip_type"
    "$INSTALL_COMMON_PARSER_PATH" --uninstall ${shell_options_} ${custom_options_} \
        "${install_type}" "${input_install_path}" "${FILELIST_PATH}" "${feature_type}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Uninstall ${PACKAGE_NAME} files failed."
        return 1
    fi

    removeVersionInfo
    log $LEVEL_INFO "Uninstall ${PACKAGE_NAME} files succeed in ${install_dir}!"
}

uninstallProfiling() {
    profiler_install_shell="${install_dir}/${PACKAGE_NAME}/script/install_msprof_fitter.sh"
    if [ ! -f "${profiler_install_shell}" ]; then
        return 0
    fi

    if [ $quiet = y ]; then
        bash "$profiler_install_shell" --uninstall --quiet
    else
        bash "$profiler_install_shell" --uninstall
    fi
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Uninstall profiling failed!"
        return 1
    fi

    log $LEVEL_INFO "Uninstall profiling succeed!"
    return 0
}

uninstallModule() {
    local shell_info="${install_dir}/${PACKAGE_NAME}/script/shells.info"
    if [ ! -f "${shell_info}" ]; then
        return 0
    fi

    local param_quiet=""
    if [ ! x$quiet = "x" ] && [ $quiet = y ]; then
        param_quiet="--quiet"
    fi

    shell_array=$(readShellInfo "${shell_info}" "[uninstall]" "[end]")
    for item in ${shell_array[@]}; do
        local shell_path="${install_dir}/${PACKAGE_NAME}/script/${item}"
        if [ ! -f $shell_path ]; then
            log_and_print $LEVEL_WARN "$shell_path not exist."
            continue
        fi
        "${shell_path}" ${param_quiet}
        if [ $? -ne 0 ];then
            log_and_print $LEVEL_ERROR "Execute $shell_path failed, please check and retry!"
            return 1
        fi
    done
    return 0
}

# load common shell
if [ -f "${COMMON_SHELL_PATH}" ]; then
    source "${COMMON_SHELL_PATH}"
fi

installInfo=$install_dir/$PACKAGE_NAME/$INSTALL_INFO_FILE # install config
if [ ! -f "${installInfo}" ];then
    log ${LEVEL_WARN} "${installInfo} does not exist."
fi
username=$(getInstallParam "UserName" "${installInfo}")
usergroup=$(getInstallParam "UserGroup" "${installInfo}")

checkGroup ${usergroup} ${username}
if [ $? -ne 0 ]; then
    usergroup=$DEFAULT_USERGROUP
fi
feature_type=$(getInstallParam "Feature_Type" "${installInfo}")
if [ -z ${feature_type} ]; then
    feature_type="all"
fi

chip_type=$(getInstallParam "Chip_Type" "${installInfo}")
if [ -z ${chip_type} ]; then
    chip_type="all"
fi

uninstallProfiling
if [ $? -ne 0 ];then
    exit 1
fi

uninstallModule
if [ $? -ne 0 ]; then
    exit 1
fi

uninstallTool
if [ $? -ne 0 ];then
    exit 1
fi

exit 0

