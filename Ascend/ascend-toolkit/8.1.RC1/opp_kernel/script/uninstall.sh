#!/bin/bash
_curr_path=$(dirname $(readlink -f $0))
OPP_KERNEL_COMMON="${_curr_path}""/opp_kernel_common.sh"
targetdir=$(readlink -f "${_curr_path}/../")
ascend_version_dir=$(readlink -f "${_curr_path}/../../")
logFile="/var/log/ascend_seclog/ascend_install.log"
installInfo="${_curr_path}/../ascend_install.info"

. "${OPP_KERNEL_COMMON}"

# get the path of this shell
currentPath=$(dirname $(readlink -f "${0}"))
areaNum=`echo "$currentPath" | awk -F'/' '{print NF}'`
# uninstallPath=`echo "$currentPath" | cut -d'/' -f1-$(expr $areaNum - 3)`

# init arch 
architecture=$(uname -m)
architectureDir="${architecture}-linux"

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

deleteLeft(){
    local path="$1/opp/built-in/op_impl/ai_core/tbe/kernel"
    local tmp
    if [ -d $path ];then
        while [ "$path" != "$1" ]
        do
            tmp=$(dirname "${path}")
            local count=`ls $path | wc -l`
            if [ "$count" -ne "0" ];then
                return 1
            fi
            rm -rf  "$path" > /dev/null 2>&1
            path=$tmp
        done
    fi
    return 0
}

deleteopp(){
    if [[ -s "$1" && -d "$1" ]];then
       for file in `ls -a "$1"`
       do
         if [ -f "$1/$file" ];then
           return 1
         fi
        if test -d "$1/$file";then
            if [[ "$file" != '.' && "$file" != '..' ]];then
                   return 1
            fi
        fi
       done
       rm -rf "$1"
    fi
}

deleteInstallpath(){
    local tmpinstallpath=$1
    local trueend=1
    if [[ -s "$tmpinstallpath" && -d "$tmpinstallpath" ]];then
        for file in `ls -a "$tmpinstallpath"`; do
            rm -rf "$tmpinstallpath" > /dev/null 2>&1
        done
    fi
}


LOG_OPERATION_INSTALL="Install"
LOG_OPERATION_UPGRADE="Upgrade"
LOG_OPERATION_UNINSTALL="Uninstall"
LOG_LEVEL_SUGGESTION="SUGGESTION"
LOG_LEVEL_MINOR="MINOR"
LOG_LEVEL_MAJOR="MAJOR"
LOG_LEVEL_UNKNOWN="UNKNOWN"
LOG_RESULT_SUCCESS="success"
LOG_RESULT_FAILED="failed"
RUN_DIR="$(echo $2 | cut -d'-' -f 3)"
operation="${LOG_OPERATION_UNINSTALL}"
runfilename="Oppkernel"
LOG_RESULT_SUCCESS="success"
LOG_RESULT_FAILED="failed"
start_time=$(date +"%Y-%m-%d %H:%M:%S")
default_normal_username=$(id -nu)
default_narmal_usergroup=$(id -ng)

# operator log
logOperation() {
    local operation="$1"
    local start_time="$2"
    local runfilename="$3"
    local result="$4"
    local installmode="$5"
    local cmdlist="$6"
    local level

    if [ "${operation}" = "${LOG_OPERATION_INSTALL}" ]; then
        level="${LOG_LEVEL_SUGGESTION}"
    elif [ "${operation}" = "${LOG_OPERATION_UPGRADE}" ]; then
        level="${LOG_LEVEL_MINOR}"
        installmode=""
    elif [ "${operation}" = "${LOG_OPERATION_UNINSTALL}" ]; then
        level="${LOG_LEVEL_MAJOR}"
    else
        level="${LOG_LEVEL_UNKNOWN}"
    fi

    if [ ! -d "${OPERATION_LOGDIR}" ]; then
        mkdir -p ${OPERATION_LOGDIR}
        chmod 750 ${OPERATION_LOGDIR} > /dev/null 2>&1
    fi

    if [ ! -f "${OPERATION_LOGPATH}" ]; then
        touch ${OPERATION_LOGPATH}
        chmod 640 ${OPERATION_LOGPATH} > /dev/null 2>&1
    fi

    if [ "${installmode}" = "full" ] || [ "${installmode}" = "run" ] || [ "${installmode}" = "devel" ]; then
        echo "${operation} ${level} ${default_normal_username} ${start_time} 127.0.0.1 ${runfilename} ${result}"\
            "installmode=${installmode}; cmdlist=${cmdlist}." >> ${OPERATION_LOGPATH}
    else
        echo "${operation} ${level} ${default_normal_username} ${start_time} 127.0.0.1 ${runfilename} ${result}"\
            "installmode=${installmode}; cmdlist=${cmdlist}." >> ${OPERATION_LOGPATH}
    fi
}
uninstallPath=$(getInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "${installInfo}")

# 创建文件夹
createLogFolder() {
    if [ ! -d $log_dir ]; then
        mkdir -p $log_dir
    fi
    if [ $(id -u) -ne 0 ]; then
        chmod  740  $log_dir > /dev/null 2>&1
    else
        chmod 750 $log_dir > /dev/null 2>&1
    fi
}

# change log file mode
changeLogMode() {
    if [ ! -f $logFile ]; then
        touch $logFile
    fi
    chmod 640 $logFile > /dev/null 2>&1
}

startLog() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    logPrint "[INFO]: Start time: ${cur_date}"
    log "[INFO]: Start time: ${cur_date}"
}

exitLog() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    logPrint "[INFO]: End time: ${cur_date}"
    log "[INFO]: End time: ${cur_date}"
    exit $1
}

showPath() {
    logPrint "[INFO]: uninstall path: ${targetdir}/opp/built-in/op_impl/ai_core/tbe/kernel"
    log "[INFO]: uninstall path: ${targetdir}/opp/built-in/op_impl/ai_core/tbe/kernel"
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

printOppkernelFile(){
    if [[ -s "$1/opp/built-in/op_impl/ai_core/tbe/kernel" && -d "$1/opp/built-in/op_impl/ai_core/tbe/kernel" ]];then
        log "[INFO]: Some files generated by user are not cleared, if necessary, manually clear them, get details in log"
        logPrint "[INFO]: Some files generated by user are not cleared, if necessary, manually clear them, get details in $logFile"
        for file in `ls -a "$1"/opp/built-in/op_impl/ai_core/tbe/kernel`
        do
            if [ -f "$1/opp/built-in/op_impl/ai_core/tbe/kernel/$file" ];then
            log "[INFO]: $1/opp/built-in/op_impl/ai_core/tbe/kernel/$file"
            fi
        if test -d "$1/opp/built-in/op_impl/ai_core/tbe/kernel/$file";then
            if [[ "$file" != '.' && "$file" != '..' ]];then
                printFile "$1/opp/built-in/op_impl/ai_core/tbe/kernel/$file"
            fi
        fi
        done
    fi
}

printFile(){
    local status=0
    for file in `ls -a "$1"`
    do
        if [ -f "$1/$file" ];then
            status=1
            log "[INFO]: $1/$file"
        fi
        if test -d "$1/$file";then
            if [[ "$file" != '.' && "$file" != '..' ]];then
                status=2
                printFile "$1/$file"
            fi
        fi
    done
    if [ $status == 0 ];then
        log "[INFO]: $1"
    fi
}

oppkernelManualRemove() {
    version_info="$uninstallPath/opp_kernel/version.info"
    "$uninstallPath"/opp_kernel/script/install_common_parser.sh --package="opp_kernel" --uninstall \
        --version-file="${version_info}" \
        --recreate-softlink \
        --username="unknown" --usergroup="unknown" --arch=${architecture} full \
        "$(dirname $uninstallPath)" \
        "$uninstallPath/opp_kernel/script/filelist.csv"
    if [ $? -ne 0 ]; then
        log "[ERROR]: uninstall "$uninstallPath"/opp/built-in/op_impl/ai_core/tbe/kernel opp kernel fail."
    fi
    printOppkernelFile "$uninstallPath"
    log "[INFO]: Opp_kernel package uninstalled successfully! Uninstallation takes effect immediately."
    logPrint "[INFO]: Opp_kernel package uninstalled successfully! Uninstallation takes effect immediately."
    return 0
}

tmppath=""
getAbsolutePath(){
    tmppath="${1}"
    local current_path="${PWD}"
    flag1=$(echo "${tmppath:0:3}" |grep "\./")''$(echo "$tmppath" |grep "~/")''$(echo "$tmppath" |grep "/\.")
    flag2=$(echo "${tmppath:0:1}" |grep "/")
    if [ -z $flag1  ] && [  -z $flag2  ]; then
        tmppath=""
        return 1
    fi
    if [ ! -z $flag1 ];then
        cd "$RUN_DIR" >& /dev/null
        eval "cd ${tmppath}" >& /dev/null
        if [ $? -ne 0 ]; then
             tmppath=""
             cd "${current_path}" >& /dev/null
             return 1
        else
           tmppath="${PWD}"
           cd "${current_path}" >& /dev/null
           return 0
        fi
    fi
}

removesoftlink() {
    path="$1"
    if [ -L "$1" ]; then
        rm -fr ${path}
        return 0
    else
        return 1
    fi
}


emptyDirRemove() {
    if [ -f $installInfo ]; then
        rm -fr "$installInfo"
    fi
    if [ -z "$(ls -A $ascend_version_dir)" ]; then
        rm -fr "$ascend_version_dir"
    fi
    if [ -z "$(ls -A $(dirname $ascend_version_dir))" ]; then
        rm -fr "$(dirname $ascend_version_dir)"
    fi
    remove_cann_ops ${uninstallPath}
}

# start
quiet_install=n
createLogFolder
changeLogMode

while true
do
    case "$1" in
        --help | -h)
            echo "--help | -h     print help message"
            echo "--quiet         quiet uninstall mode, skip human-computer interactions"
            echo "--install-path=<path>             Specify the path to install opp_kernels module"
            exit 0
            ;;
        --quiet)
            quiet_install=y
            shift
            ;;
        --install-path=*)
            input_install_userpath=y
            temp_path=`echo "$1" | cut -d"=" -f2- `
            slashes_num=$( echo "${temp_path}" | grep -o '/' | wc -l )
            # 去除指定安装目录后所有的 "/"
            if [ $slashes_num -gt 1 ];then
                uninstallPath=`echo "${temp_path}" | sed "s/\/*$//g"`
            else
                uninstallPath="${temp_path}"
            fi
            shift
            ;;
        -*)
            echo "unsupported parameters : $1"
            log "[ERROR]: unsupported parameters : $1"
            exit 0
            ;;
        *)
            if [ "xx${1}" = "xx" ];then
                break
            else
                echo "unsupported parameters : $1"
                log "[ERROR]: unsupported parameters : $1"
                exit 0
            fi
            ;;
    esac
done

startLog
logPrint "[INFO]: UninstallLogFile: ${logFile}"
uninstallPath=$(getInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "${installInfo}")
if [ "${input_install_userpath}" = "y" ];then
        getAbsolutePath ${uninstallPath}
        if [ $? -eq 0 ];then
            updateTargetDir "$uninstallPath"
            installInfo="$uninstallPath/opp_kernel/ascend_install.info"
            installmode=$(getInstallParam "OPP_KERNEL_INSTALL_TYPE" "${installInfo}")
            all_parma="--uninstall --install-path=$uninstallPath"

            oppkernelManualRemove
            if [ $? -ne 0 ];then
            colorPrint "[ERROR]:\033[31m remove opp_kernel failed, details in : $logFile \033[0m"
            log "[ERROR]: remove opp_kernel failed, details in : $logFile"
            exitLog 1
            fi
            deleteopp "${uninstallPath}/opp/built-in/op_impl/ai_core/tbe/kernel"
            rm -fr "${uninstallPath}/opp_kernel" > /dev/null 2>&1
            # if dir is empty then remove it
            emptyDirRemove
            if [ $? -ne 0 ]; then
                log "[ERROR]: empty dirs remove failed"
                exit 1
            fi
            logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_SUCCESS}" "$installmode" "$all_parma"
        else
            log "[ERROR]: target path($uninstallPath) is wrong, uninstall failed"
            logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_FAILED}" "$installmode" "$all_parma"
            exitLog 1
        fi
else
    SHELL_PATH="$(dirname "$(readlink -f $0)")" # 脚本目录
    UNINSTALL_SHELL="${SHELL_PATH}/run_opp_kernel_uninstall.sh" # 卸载脚本路径
    INSTAll_PATH_PARAM="$(dirname "${SHELL_PATH}")"
    installInfo="$INSTAll_PATH_PARAM/ascend_install.info"
    installmode=$(getInstallParam "OPP_KERNEL_INSTALL_TYPE" "${installInfo}")
    all_parma="--uninstall --install-path=$uninstallPath"

    oppkernelManualRemove
    if [ $? -ne 0 ];then
    colorPrint "[ERROR]:\033[31m remove opp_kernel failed, details in : $logFile \033[0m"
    log "[ERROR]: remove opp_kernel failed, details in : $logFile"
    logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_FAILED}" "$installmode" "$all_parma"
    exitLog 1
    fi
    deleteopp "${uninstallPath}/opp/built-in/op_impl/ai_core/tbe/kernel"
    rm -fr "${uninstallPath}/opp_kernel" > /dev/null 2>&1
    deleteInstallpath ${uninstallPath}/opp/built-in/op_impl/ai_core/tbe/kernel
    # if dir is empty then remove it
    emptyDirRemove
    if [ $? -ne 0 ]; then
        log "[ERROR]: empty dirs remove failed"
        exit 1
    fi
    logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_SUCCESS}" "$installmode" "$all_parma"
fi
exitLog 0
