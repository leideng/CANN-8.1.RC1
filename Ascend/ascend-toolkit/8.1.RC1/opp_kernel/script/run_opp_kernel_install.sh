#!/bin/bash
targetdir=/usr/local/Ascend
# installInfo="$1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ascend_install.info"
installInfo="$1/opp_kernel/ascend_install.info"
_curr_path=$(dirname $(readlink -f $0))
OPP_KERNEL_COMMON="${_curr_path}""/opp_kernel_common.sh"
_common_parser_file="${_curr_path}""/install_common_parser.sh"
_VERSION_INFO_FILE="${_curr_path}/../version.info"
filelist="${_curr_path}""/../filelist.csv"
kernel_scene_info="${_curr_path}""/../scene.info"
opp_scene_info="${_curr_path}""/../opp/scene.info"
ascend_file="${_curr_path}""/../opp_kernel/ascend910b"
ascend_config="${_curr_path}""/../opp_kernel/config"
_COMMON_INC_FILE="${_curr_path}""/common_func.inc"
VERSION_COMPAT_FUNC_PATH="${_curr_path}/version_compatiable.inc"
common_func_v2_path="${_curr_path}/common_func_v2.inc"
version_cfg_path="${_curr_path}/version_cfg.inc"
. "${VERSION_COMPAT_FUNC_PATH}"
. "${_COMMON_INC_FILE}"
. "${common_func_v2_path}"
. "${version_cfg_path}"
. "${OPP_KERNEL_COMMON}"

get_version_dir "pkg_version_dir" "$_VERSION_INFO_FILE"

#get installinfo
getInstallParam() {
    local _key="$1"
    local _file="$2"
    local _param
    if [ ! -f "${_file}" ];then
        exit 1
    fi
    install_info_key_array=("OPP_KERNEL_INSTALL_TYPE" "OPP_KERNEL_FEATURE_TYPE" "OPP_KERNEL_CURRENT_FEATURE" "OPP_KERNEL_INSTALL_PATH_PARAM" "USERNAME" "USERGROUP" "ASCEND_OPP_KERNEL_PATH")
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
            break
        fi
    done
    echo "${_param}"
}

# . $installInfo
UserName=$(getInstallParam "USERNAME" "${installInfo}")
UserGroup=$(getInstallParam "USERGROUP" "${installInfo}")
username="$UserName"
usergroup="$UserGroup"
if [ "$username" = "" ]; then
    username=$(id -un)
    usergroup=$(id -gn)
fi

FeatureList=$(getInstallParam "OPP_KERNEL_CURRENT_FEATURE" "${installInfo}")
featurelist="$FeatureList"

# init arch 
architecture=$(uname -m)
architectureDir="${architecture}-linux"

showPath() {
    log "[INFO]: target path : $targetdir"
}

updateTargetDir() {
    if [ ! -z "$1" ] && [ -d "$1" ];then
        targetdir="$1"
        targetdir="${targetdir%*/}"
    fi
}
createInstallDir() {
    local path=$1
    local user_and_group=$2

    if [ $(id -u) -eq 0 ]; then
        user_and_group="root:root"
        permission=755
    else
        permission=750
    fi

    if [ x"${path}" = "x" ]; then
        log "[WARNING]:dir path is empty"
        return 1
    fi
    mkdir -p "${path}"
    if [ $? -ne 0 ]; then
        log "[WARNING]:create path="${path}" failed."
        return 1
    fi
    chmod -R $permission "${path}" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log "[WARNING}:chmod path="${path}" $permission failed."
        return 1
    fi
    chown -f $user_and_group "${path}"
    if [ $? -ne 0 ]; then
        log "[WARNING]:chown path="${path}" $user_and_group failed."
        return 1
    fi
}

oppKernelInstall() {
    is_for_all=$2
    if [ "${is_for_all}" = "y" ]; then
        is_install_all="--install_for_all"
    else
        is_install_all=""
    fi
    local mod_num=""
    mod_num=$(stat -c %a "$1/opp" > /dev/null 2>&1)
    local ascend_mod_num=""
    ascend_mod_num=$(stat -c %a "$1/opp" > /dev/null 2>&1)
    logPrint "[INFO]: Start to install opp kernel"

    # delete old version kernel uninstall script for compatibility
    if [ -f "$1/opp/op_impl/built-in/ai_core/tbe/kernel/scripts/uninstall.sh" ]; then
        rm -rf $1/opp/op_impl/built-in/ai_core/tbe/kernel/scripts
    fi
    
    if [ -f "$1/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/uninstall.sh" ]; then
        rm -rf $1/opp/built-in/op_impl/ai_core/tbe/kernel/scripts
    fi

    # replace the arch variable in scene.info
    actualArch="arch=${architecture}"
    if [ -f $kernel_scene_info ]; then
        sed -i "s/arch=.*/${actualArch}/" $kernel_scene_info
    fi
    if [ -f $opp_scene_info ]; then
        sed -i "s/arch=.*/${actualArch}/" $opp_scene_info
    fi

    # replace the symbolic variable in filelist.csv
    newStr="${architectureDir}"
    oldStr="\$(TARGET_ENV)"
    sed -i 's#'''$oldStr'''#'''$newStr'''#g' ${filelist}

    # indicates whether to install incrementally
    if [ -f "$1/opp_kernel/version.info" ]; then
        sh $_common_parser_file --package="opp_kernel" --install ${is_install_all} --version-file="${_VERSION_INFO_FILE}" \
            --username="${username}" --usergroup="${usergroup}" --set-cann-uninstall --arch=${architecture} --increment \
            --feature="${featurelist}" "${installMode}" "$(dirname $1)" ${filelist}
        if [ $? -ne 0 ];then
            log "[ERROR]: opp kernel increment install failed"
            return 1
        fi
    else    
        sh $_common_parser_file --package="opp_kernel" --install ${is_install_all} --version-file="${_VERSION_INFO_FILE}" \
            --username="${username}" --usergroup="${usergroup}" --set-cann-uninstall --arch=${architecture} \
            --feature="${featurelist}" "${installMode}" "$(dirname $1)" ${filelist}
        if [ $? -ne 0 ];then
            log "[ERROR]: opp kernel install failed"
            return 1
        fi
    fi

    if [ "${is_for_all}" = "y" ]; then
        chmod -R 755 "$1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" > /dev/null 2>&1
    else
        chmod -R $ascend_mod_num "$1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" > /dev/null 2>&1
    fi

    log "[INFO]: opp kernel install successfully"
    return 0
}

createsoftlink() {
    _src_path="$1"
    _dst_path="$2"
    if [ -L "$2" ]; then
        return 0
    fi
    ln -s "${_src_path}" "${_dst_path}" 2> /dev/null
    if [ "$?" != "0" ]; then
        return 1
    else
        return 0
    fi
}
 
# start!
install_Path_Param="$1"
quiet_install="$2"
hot_reset_check="$3"
install_all="$4"
installMode="$5"
updateTargetDir "$install_Path_Param"


showPath
oppKernelInstall "$targetdir" "${install_all}"
if [ $? -ne 0 ];then
    log "[ERROR]: opp kernel install failed"
    exit 1
fi


# delete op_proto and op_tiling of different architectures
logPrint "[INFO]: Start delete different architectures."
deleteDiffernetArchs $1
return_code=$?
if [ ${return_code} -eq 0 ]; then
    logPrint "[Info]: Delete different architectures successfully!"
elif [ ${return_code} -eq 3 ]; then
    logPrint "[WARNING]: op_proto and op_tiling source file does not exist!"
else
    logPrint "[ERROR]: Delete different architectures failed!"
    exit 1
fi

logPrint "[INFO]: installPercentage: 100%"
log "[INFO]: installPercentage: 100%"
exit 0
