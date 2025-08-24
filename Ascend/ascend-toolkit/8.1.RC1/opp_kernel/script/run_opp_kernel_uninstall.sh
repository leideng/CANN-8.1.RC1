#!/bin/bash
targetdir=/usr/local/Ascend
logFile="/var/log/ascend_seclog/ascend_install.log"
installInfo="$1/opp_kernel/ascend_install.info"
_CURR_PATH=$(dirname $(readlink -f $0))
_FILELIST_FILE="${_CURR_PATH}""/filelist.csv"
OPP_KERNEL_COMMON="${_CURR_PATH}""/opp_kernel_common.sh"
_COMMON_PARSER_FILE="${_CURR_PATH}""/install_common_parser.sh"

. "${OPP_KERNEL_COMMON}"

# get the path of this shell
currentPath=$(dirname $(readlink -f "${0}"))
areaNum=`echo "$currentPath" | awk -F'/' '{print NF}'`

# init arch 
architecture=$(uname -m)
architectureDir="${architecture}-linux"

getInstallParam() {
    local _key="$1"
    local _file="$2"
    local _param
    if [ ! -f "${_file}" ];then
        exit 1
    fi
    install_info_key_array=("OPP_KERNEL_INSTALL_TYPE" "OPP_KERNEL_INSTALL_PATH_PARAM" "USERNAME" "USERGROUP")
    for key_param in "${install_info_key_array[@]}"; do
        if [ ${key_param} == ${_key} ]; then
            _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
            break
        fi
    done
    echo "${_param}"
}
uninstallPath=$(getInstallParam "OPP_KERNEL_INSTALL_TYPE" "${installInfo}")
target_username=$(getInstallParam "USERNAME" "${installInfo}")
target_usergroup=$(getInstallParam "USERGROUP" "${installInfo}")

showPath() {
    log "[INFO]: uninstall path: $targetdir"
}

updateTargetDir() {
    if [ ! -z "$1" ] && [ -d "$1" ]; then
        targetdir="$1"
        targetdir="${targetdir%*/}"
    else
        log "[ERROR]: target path($1) is wrong, uninstall failed"
        return 1
    fi

    showPath
    return 0
}

removeopapisoftlink() {
    path="$1"
    if [ -L "$1" ]; then
        rm -fr ${path}
        return 0
    else
        return 1
    fi
}


oppkernelRemove() {
    version_info="$1/opp_kernel/version.info"
    sh "${_COMMON_PARSER_FILE}" --package="opp_kernel" --uninstall --version-file="${version_info}" \
        --username="${target_username}" --usergroup="${target_usergroup}" --arch=${architecture} \
        --recreate-softlink \
        "full" "$(dirname $1)" "${_FILELIST_FILE}"
    if [ $? -ne 0 ]; then
        log "[ERROR]: uninstall "$1"/opp/built-in/op_impl/ai_core/tbe/kernel opp kernel fail."
    fi
    return 0
}

emptyDirRemove() {
    if [ -f $installInfo ]; then
        rm -fr "$installInfo"
    fi
    if [ -z "$(ls -A $targetdir/opp_kernel/)" ]; then
        rm -fr "$targetdir/opp_kernel/"
    fi
    if [ -z "$(ls -A $targetdir)" ]; then
        rm -fr "$targetdir"
    fi
    if [ -z "$(ls -A $(dirname $targetdir))" ]; then
        rm -fr "$(dirname $targetdir)"
    fi
    remove_cann_ops ${targetdir}
}

# start
if [ ! "x${1}" = "x" ];then
    uninstallPath="$1"
fi
installMode="$2"
updateTargetDir "$uninstallPath"

# for run package uninstall
if [ ! -d "$targetdir"/opp/built-in/op_impl/ai_core/tbe/kernel ];then
    if [[ $installMode = "uninstall" ]];then
        logPrint "[ERROR]: failed to find opp_kernel install path, run_opp_kernel_uninstall failed!"
        log "[ERROR]: failed to find opp_kernel install path, run_opp_kernel_uninstall failed!"
        exit 1
    else
        logPrint "[INFO]: $targetdir/opp/built-in/op_impl/ai_core/tbe/kernel is not existed, no need to delete!"
        log "[INFO]: $targetdir/opp/built-in/op_impl/ai_core/tbe/kernel is not existed, no need to delete!"
        exit 0;
    fi
fi


oppkernelRemove "$targetdir"
if [ $? -ne 0 ]; then
    log "[ERROR]: execute install_common_parser.sh remove failed"
    exit 1
fi

# if dir is empty then remove it
emptyDirRemove
if [ $? -ne 0 ]; then
    log "[ERROR]: empty dirs remove failed"
    exit 1
fi
exit 0
