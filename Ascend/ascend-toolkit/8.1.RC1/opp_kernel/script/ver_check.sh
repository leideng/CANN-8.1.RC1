#!/bin/bash
logFile="/var/log/ascend_seclog/ascend_install.log"
_curr_path=$(dirname $(readlink -f $0))
opp_kernel_version="${_curr_path}""/../version.info"
installInfo="$1/opp_kernel/ascend_install.info"
if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
logFile="${log_dir}/ascend_install.log"
#get installinfo
getInstallParam() {
    local _key="$1"
    local _file="$2"
    local _param
    if [ ! -f "${_file}" ];then
        exit 1
    fi
    install_info_key_array=("OPP_KERNEL_INSTALL_TYPE" "OPP_KERNEL_INSTALL_PATH_PARAM" "USERNAME" "USERGROUP")
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
            break
        fi
    done
    echo "${_param}"
}

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Opp_Kernel] [$cur_date] "$1 >> $logFile
}
logPrint() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Opp_Kernel] [$cur_date] ""$1"
}

installCheckVersion(){
    if [ -f "$installInfo" ]; then
        # . $installInfo
        opp_Install_Path_Param=$(getInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "${installInfo}")
	req_version="${opp_Install_Path_Param}""/opp/version.info"
        checkFunc="./common_func.inc"
        req_pkg=opp
        if [ -f "$checkFunc" ]; then
            . $checkFunc
            if [ -f "$opp_kernel_version" ]; then
                if [ -f "$req_version" ]; then
                    check_pkg_ver_deps "$opp_kernel_version" "$req_pkg" "$req_version"
                    if [ $VerCheckStatus == SUCC ]; then
                        logPrint "[INFO]: version meet requirement"
                        log "[INFO]: version meet requirement"
                        exit 0
                    else
                        logPrint "[WARNING]: opp kernel version is lower than required"
                        log "[WARNING]: opp kernel version is lower than required"
                        exit 1
                    fi
                else
                    logPrint "[WARNING]: opp version.info not exist"
                    log "[WARNING]: opp version.info not exist"
                    exit 1
                fi
            else
                logPrint "[WARNING]: opp kernel version.info not exist"
                log "[WARNING]: opp kernel version.info not exist"
                exit 1
            fi
        else
            logPrint "[WARNING]: common_func.inc not exist"
            log "[WARNING]: common_func.inc not exist"
            exit 1
        fi
    else
        logPrint "[WARNING]: ascend_install.info not exist"
        log "[WARNING]: ascend_install.info not exist"
        exit 1
    fi
}

installCheckVersion
