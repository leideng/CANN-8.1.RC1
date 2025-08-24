#!/bin/bash
default_root_dir="/usr/local/Ascend"
default_normal_dir="${HOME}/Ascend"
default_normal_username=$(id -nu)
default_narmal_usergroup=$(id -ng)
installInfo="./opp_kernel/ascend_install.info"
installPath="/lib/davinci.conf"
username=$(id -un)
usergroup=$(id -gn)
input_install_path=""
usageFile="./opp/script/opp_kernel_help.info"
_CURR_PATH=$(dirname $(readlink -f $0))
LOG_OPERATION_INSTALL="Install"
LOG_OPERATION_UPGRADE="Upgrade"
LOG_OPERATION_UNINSTALL="Uninstall"
LOG_LEVEL_SUGGESTION="SUGGESTION"
LOG_LEVEL_MINOR="MINOR"
LOG_LEVEL_MAJOR="MAJOR"
LOG_LEVEL_UNKNOWN="UNKNOWN"
LOG_RESULT_SUCCESS="success"
LOG_RESULT_FAILED="failed"
OPP_KERNEL_COMMON="${_CURR_PATH}""/opp_kernel_common.sh"
_FILELIST_FILE="${_CURR_PATH}""/../filelist.csv"
_COMMON_INC_FILE="${_CURR_PATH}""/common_func.inc"
VERSION_COMPAT_FUNC_PATH="${_CURR_PATH}/version_compatiable.inc"
_VERSION_INFO_FILE="${_CURR_PATH}""/../version.info"
_SCENE_INFO_FILE="${_CURR_PATH}""/../scene.info"
common_func_v2_path="${_CURR_PATH}/common_func_v2.inc"
version_cfg_path="${_CURR_PATH}/version_cfg.inc"
opp_env_dir=${ASCEND_OPP_PATH}
opp_env_dir_cut=${opp_env_dir%/*}
targetdir=${opp_env_dir_cut%/*}

. "${VERSION_COMPAT_FUNC_PATH}"
. "${_COMMON_INC_FILE}"
. "${common_func_v2_path}"
. "${version_cfg_path}"
. "${OPP_KERNEL_COMMON}"

# init arch 
architecture=$(uname -m)
architectureDir="${architecture}-linux"

RUN_DIR="$(echo $2 | cut -d'-' -f 3)"

. ${_COMMON_INC_FILE}

if [ $(id -u) -ne 0 ]; then
    username="${default_normal_username}"
    usergroup="${default_narmal_usergroup}"
    default_root_dir="${default_normal_dir}"
fi

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
        touch $logFile > /dev/null 2>&1
    fi
    chmod 640 $logFile > /dev/null 2>&1
}

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

updateInstallParam() {
    local _key="$1"
    local _val="$2"
    local _file=$3
    local _param

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    install_info_key_array=("OPP_KERNEL_INSTALL_TYPE" "OPP_KERNEL_FEATURE_TYPE" "OPP_KERNEL_CURRENT_FEATURE" "OPP_KERNEL_INSTALL_PATH_PARAM" "USERNAME" "USERGROUP" "ASCEND_OPP_KERNEL_PATH")
    for key_param in "${install_info_key_array[@]}"; do
        if [ ${key_param} == ${_key} ]; then
            if [ "$key_param" == "ASCEND_OPP_KERNEL_PATH" ];then
                sed -i "/export ${_key}=/d" "${_file}"
                echo "export ${_key}=${_val}" >> "${_file}"
            else
                sed -i "/${_key}=/d" "${_file}"
                echo "${_key}=${_val}" >> "${_file}"
            fi
        fi
    done
}

writeSource(){
    logPrint "[INFO]: Using requirements: when opp kernel module install finished or before you run the opp kernel module, execute the command"
    log "[INFO]: Using requirements: when opp kernel module install finished or before you run the opp kernel module, execute the command"
    echo "[ export ASCEND_OPP_KERNEL_PATH=${1} ] to set the environment path."
    log "[ export ASCEND_OPP_KERNEL_PATH=${1} ] to set the environment path."
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
    while [ $trueend -eq 1 ]
    do
        if [[ -s "$tmpinstallpath" && -d "$tmpinstallpath" ]];then
        for file in `ls -a "$tmpinstallpath"`
        do
                if [ -f "$tmpinstallpath/$file" ];then
                    trueend=0
                    return 1
                fi
                if test -d "$tmpinstallpath/$file";then
                    if [[ "$file" != '.' && "$file" != '..' ]];then
                        trueend=0
                        return 1
                    fi
                fi
        done
        rm -rf "$tmpinstallpath" > /dev/null 2>&1
        tmpinstallpath=$(dirname "${tmpinstallpath}")
        fi
    done
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
        log "[INFO]: ""$1"
    fi
}

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
            "cmdlist=${cmdlist}." >> ${OPERATION_LOGPATH}
    fi
}

# print help.info file
paramUsage()
{
    logPrint "[INFO]: Usage: ./${runfilename}.run -- arg1 [arg2 ...]"
    cat ${usageFile}
    exitLog 1
}

# 开始安装前打印开始信息
startLog() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    logPrint "[INFO]: Start time: ${cur_date}"
    log "[INFO]: Start time: ${cur_date}"
    logPrint "[INFO]: LogFile: ${logFile}"
    logPrint "[INFO]: OperationLogFile: ${OPERATION_LOGPATH}"
    log "[INFO]: LogFile: ${logFile}"
    log "[INFO]: OperationLogFile: ${OPERATION_LOGPATH}"
    log "[INFO]: InputParams: $all_parma"
}

# 退出前打印结束信息
exitLog() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    logPrint "[INFO]: End time: ${cur_date}"
    log "[INFO]: End time: ${cur_date}"
    exit $1
}

# 检查用户是否存在
checkUser(){
    ret=`cat /etc/passwd | cut -f1 -d':' | grep -w "$1" -c`
    if [ $ret -le 0 ]; then
        return 1
    else
        return 0
    fi
}

# 校验user和group的关联关系
checkGroup(){
    group_user_related=`groups "$2"|awk -F":" '{print $2}'|grep -w "$1"`
    if [ "${group_user_related}x" != "x" ];then
        return 0
    else
        return 1
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
    chown -RPf $user_and_group "${path}"
    if [ $? -ne 0 ]; then
        log "[WARNING]:chown path="${path}" $user_and_group failed."
        return 1
    fi
}

# 检查是否满足安装条件
checkInstallCondition() {
    if [ $(id -u) -ne 0 ]; then
        username=$default_normal_username
        usergroup=$default_narmal_usergroup
        # createInstallDir "${Install_Path_Param}" "${username}:${usergroup}"
    fi

}

getUserInfo() {
    if [ -f ${installInfo} ]; then
        UserName=$(getInstallParam "USERNAME" "${installInfo}")
        UserGroup=$(getInstallParam "USERGROUP" "${installInfo}")
        if [ ${input_install_username} = y ] || [ ${input_install_usergroup} = y ]; then
            if [ "x${UserName}" = "x" ]; then
                    log "[ERROR]: ERR_NO:0x0095;ERR_DES:The user name is null"
                    logPrint "[ERROR]: ERR_NO:0x0095;ERR_DES:The user name is null"
                    exitLog 1
            else
                if [ "x${username}" != "x${UserName}" ]; then
                    log "[ERROR]: ERR_NO:0x0095;ERR_DES:The user is not the same as opp kernel installation user"
                    logPrint "[ERROR]: ERR_NO:0x0095;ERR_DES:The user is not the same as opp kernel installation user"
                    exitLog 1
                else
                    username="${UserName}"
                fi
            fi

            if [ "x${UserGroup}" = "x" ]; then
                log "[ERROR]: ERR_NO:0x0095;ERR_DES:The user group is null"
                logPrint "[ERROR]: ERR_NO:0x0095;ERR_DES:The user group is null"
                exitLog 1
            else
                if [ "x${usergroup}" != "x${UserGroup}" ]; then
                    log "[ERROR]: ERR_NO:0x0095;ERR_DES:The group is not the same as opp kernel installation group"
                    logPrint "[ERROR]: ERR_NO:0x0095;ERR_DES:The group is not the same as opp kernel installation group"
                    exitLog 1
                else
                    usergroup="${UserGroup}"
                fi
            fi
        fi

        if [ ${input_install_username} = n ] || [ ${input_install_usergroup} = n ]; then
            if [ "x${UserName}" != "x" ]; then
                username="${UserName}"
            fi
            if [ "x${UserGroup}" != "x" ]; then
                usergroup="${UserGroup}"
            fi
        fi
    fi
}


# 获取安装目录下的完整版本号  version2
getVersionInstalled() {
    local version1=""
    local version2=""
    if [ -f "$1"/version.info ]; then
        . "$1"/version.info
        version1=$Version
        version2=$version_dir
    fi
    local version_res=("$version1" "$version2")
    echo "${version_res[@]}"
}

# 获取run包中的完整版本号  version1
getVersionInRunFile() {
    local version1=""
    local version2=""
    if [ -f ./version.info ]; then
        . ./version.info
        version1=${Version}
        version2=${version_dir}
    fi
    local version_res=("$version1" "$version2")
    echo "${version_res[@]}"
}

# 获取新旧的安装路径
getOldAndInputPath() {
    get_package_upgrade_install_info "installInfo" "$origin_install_path" "opp_kernel"
    if [ -f "$installInfo" ]; then
        Opp_Kernel_Install_Path_Param=$(getInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "${installInfo}")
        old_install_path="$Opp_Kernel_Install_Path_Param"
    fi
}

# 判断输入的指定路径是否存在
isValidPath() {
    local ret=0
    if [ ! -d "${input_install_path}" ]; then
        createInstallDir "${input_install_path}" "${username}":"${usergroup}"
    fi
    if [ ! -d "${input_install_path}/opp" ]; then
        createInstallDir "${input_install_path}/opp" "${username}":"${usergroup}"
    fi
    if [ $(id -u) -eq 0 ]; then
        parent_dirs_permision_check "${input_install_path}/opp" && ret=$? || ret=$?
        if [ ${quiet_install} = y ] && [ ${ret} -ne 0 ]; then
            log "[ERROR]: The given dir, or its parents, permission is invalid."
            logPrint "[ERROR]: The given dir, or its parents, permission is invalid."
            exit 1
        fi
        if [ ${ret} -ne 0 ]; then
            log "[WARNING]: You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
            logPrint "[WARNING]: You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
            while true
            do
                read yn
                if [ "$yn" = n ]; then
                    exit 1
                elif [ "$yn" = y ]; then
                    break;
                else
                    logPrint "[ERROR]: ERR_NO:0x0002;ERR_DES:input error, please input again!"
                fi
            done
        fi
    fi
    cd "$input_install_path" > /dev/null 2>&1
    if [ ! $? = 0 ]; then
        colorPrint "[ERROR]:\033[31m ERR_NO:0x0003;ERR_DES:The $username do not have the permission to access $input_install_path, please reset the directory to a right permission. \033[0m"
        log "[ERROR]: ERR_NO:0x0003;ERR_DES:The $username do not have the permission to access $input_install_path, please reset the directory to a right permission."
        cd - >/dev/null
        exitLog 1
    fi
    cd - >/dev/null
}

# 记录旧版本
logBaseVersion() {
    if [ -f ${installInfo} ];then
        # . $installInfo
        Opp_Kernel_Install_Path_Param=$(getInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "${installInfo}")
        if [ ! -z "$Opp_Kernel_Install_Path_Param" ]; then
            installed_version=$(getVersionInstalled "$OPP_KERNEL_INSTALL_PATH_PARAM"/opp_kernel)
            if [ ! "x${installed_version}" = "x" ]; then
                logPrint "[INFO]: base version is ${installed_version}."
                log "[INFO]: base version is ${installed_version}."
                return 0
            fi
        fi
    else
        logPrint "[WARNING]: base version was destroyed or not exist."
        log "[WARNING]: base version was destroyed or not exist."
    fi
}

_CURR_PATH=$(dirname $(readlink -f $0))
_INSTALL_SHELL_FILE="${_CURR_PATH}""/run_opp_kernel_install.sh"
_UPGRADE_SHELL_FILE="${_CURR_PATH}""/run_opp_kernel_upgrade.sh"
_UNINSTALL_SHELL_FILE="${_CURR_PATH}""/run_opp_kernel_uninstall.sh"
# 安装调用子脚本
installRun() {
    if [ "$1" = "install" ];then
        operation="${LOG_OPERATION_INSTALL}"
    elif [ "$1" = "upgrade" ];then
        operation="${LOG_OPERATION_UPGRADE}"
    fi
    installInfo="$input_install_path/opp_kernel/ascend_install.info"
    if [ ! -d "${input_install_path}/opp_kernel" ];then
       createInstallDir "${input_install_path}/opp_kernel" "${username}":"${usergroup}"
    fi
    if [ ! -f $installInfo ];then
        touch $installInfo
    fi
    if [ $(id -u) -eq 0 ];then
        chown -RPf "${username}:$usergroup" "$installInfo"
    fi
    chmod 600 "$installInfo" > /dev/null 2>&1
    updateInstallParam "USERNAME" "$username" "$installInfo"
    updateInstallParam "USERGROUP" "$usergroup" "$installInfo"
    updateInstallParam "OPP_KERNEL_INSTALL_TYPE" "$installMode" "$installInfo"
    updateInstallParam "OPP_KERNEL_FEATURE_TYPE" "$in_feature_all" "$installInfo"
    updateInstallParam "OPP_KERNEL_CURRENT_FEATURE" "$in_feature_new" "$installInfo"
    updateInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "$input_install_path" "$installInfo"

    bash ${_INSTALL_SHELL_FILE} "$input_install_path" "$2" "$3" "$install_all"  "${installMode}"
    ret_val=$?
    if [ $ret_val -eq 0 ];then
        if [ $(id -u) -eq 0 ]; then
            chown -RPf "root:root" "$input_install_path/script" 2> /dev/null
        else
            chown -RPf "$username:$usergroup" "$input_install_path/script" 2> /dev/null
        fi
        colorPrint "[INFO]:\033[32m Opp_kernels package installed successfully! The new version takes effect immediately. \033[0m"
        log  "[INFO]: Opp_kernel package installed successfully! The new version takes effect immediately."
        logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_SUCCESS}" "${installMode}" "${all_parma}"
    else
        uninstallRun "$1" > /dev/null 2>&1
        colorPrint "[INFO]:\033[31m Run package $1 failed, please retry after check your device and reboot! \033[0m"
        log "[INFO]: Run package $1 failed, please retry after check your device and reboot!"
        logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_FAILED}" "${installMode}" "${all_parma}"
        exitLog 1
    fi
}

# 卸载调用子脚本
uninstallRun() {
    # . $installInfo
    operation="${LOG_OPERATION_UNINSTALL}"
    if [ -z "$old_install_path" ]; then
        old_install_path="$input_install_path"
    fi
    installInfo="${old_install_path}/opp_kernel/ascend_install.info"
    logPrint "[INFO]: The old install path of opp_kernel: $old_install_path"
    uninstall_result=n
    if [ -d $old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/ ];then
        $old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/run_opp_kernel_uninstall.sh "$old_install_path" "$1" > /dev/null 2>&1
        if [ $? -ne 0 ];then
            deleteopp "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b"
            deleteopp "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel"
            uninstall_result=y
        fi
    fi

    if [ -L $old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/script ];then
        $old_install_path/opp_kernel/script/run_opp_kernel_uninstall.sh "$old_install_path" "$1" > /dev/null 2>&1
        if [ $? -ne 0 ];then
            deleteopp "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b"
            deleteopp "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel"
            uninstall_result=y
        fi
    fi

    if [ $uninstall_result = y ]; then
        if [ $1 = "uninstall" ];then
            logPrint "[INFO]: Opp_kernel package uninstalled successfully! Uninstallation takes effect immediately."
            log "[INFO]: Opp_kernel package uninstalled successfully! Uninstallation takes effect immediately."
        fi
        if [ $uninstall = y ]; then
            sed -i '/OPP_KERNEL_INSTALL_PATH_PARAM=/d' $installInfo
            sed -i '/OPP_KERNEL_INSTALL_TYPE=/d' $installInfo
            if [ ` grep -c -i "INSTALL_PATH_PARAM" $installInfo ` -eq 0 ]; then
                rm -f $installInfo
            fi
            logOperation "${operation}" "${start_time}" "${runfilename}" "${LOG_RESULT_SUCCESS}" "${installMode}" "${all_parma}"
            exitLog 0
        fi
    fi
}


judgmentpath(){
    if [ "$input_install_userpath" = "y" ];then
        if [ "x$input_install_path" != "x" ];then
            local tmp=$(relativePathToAbsolutePath "${input_install_path}")
            if [ "x$tmp" = "x" ];then
                logPrint "[ERROR]: Install path=${input_install_path} is invalid"
                log "[ERROR]: Install path=${input_install_path} is invalid"
                exitLog 1
            fi
            input_install_path="$tmp"
            check_install_path_valid "$input_install_path"
            if [ $? -ne 0 ];then
                logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:The Opp install_path $input_install_path is invalid, only [a-z,A-Z,0-9,-,_] is support!"
                log "[ERROR]: The Opp kernel install_path $input_install_path is invalid, only [a-z,A-Z,0-9,-,_] is support!"
                exitLog 1
            else
                logPrint "[INFO]: Install path=${input_install_path}"
                log "[INFO]: Install path=${input_install_path}"
            fi
        else
            logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:The Opp Kernel install_path $input_install_path is invalid."
            log "[ERROR]: ERR_NO:0x0080;ERR_DES:The Opp Kernel install_path $input_install_path is invalid."
            exitLog 1
        fi
    else
        if [ "x${targetdir}" = "x" ];then
            if [ $(id -u) -eq 0 ]; then
                input_install_path=${default_root_dir}
            else
                input_install_path=${default_normal_dir}
            fi
        else
            input_install_path=${targetdir}
        fi
    fi
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

relativePathToAbsolutePath() {
    local relative_path_="${1}"
    cd "$RUN_DIR" >& /dev/null
    eval "cd ${relative_path_}" >& /dev/null
    if [ $? -ne 0 ]; then
        if [ "$uninstall" = "y" ];then
           echo ""
        else
            cd "$RUN_DIR" >& /dev/null
            local tmprelative_path=$(dirname "${relative_path_}")
            eval "cd ${tmprelative_path}" >& /dev/null
            if [ $? -ne 0 ]; then
               echo ""
            else
                mkdir -p ${relative_path_} >& /dev/null
                if [ $? -ne 0 ];then
                    echo ""
                else
                    eval "cd ${relative_path_}" >& /dev/null
                    if [ $(id -u) -eq 0 ]; then
                        chmod 755 "${PWD}" > /dev/null 2>&1
                    else
                        chmod 750 "${PWD}" > /dev/null 2>&1
                    fi
                    echo "${PWD}"
                fi
            fi
        fi
    else
        echo "${PWD}"
    fi
}


parent_dirs_permision_check() {
    current_dir="$1" parent_dir="" short_install_dir=""
    local owner="" mod_num=""

    parent_dir=$(dirname "${current_dir}")
    short_install_dir=$(basename "${current_dir}")
    logAndPrint "[INFO]: parent_dir value is [${parent_dir}] and children_dir value is [${short_install_dir}]"

    if [ "${current_dir}"x = "/"x ]; then
        logAndPrint "[INFO]: parent dirs permission checked successfully"
        return 0
    else
        owner=$(stat -c %U "${parent_dir}"/"${short_install_dir}")
        if [ "${owner}" != "root" ]; then
            logAndPrint "[WARNING]: ${short_install_dir} permision isn't right, it should belong to root."
            return 1
        fi
        logAndPrint "[INFO]: ${short_install_dir} belongs to root."
        mod_num=$(stat -c %a "${parent_dir}"/"${short_install_dir}")
        if [ ${mod_num} -lt 755 ]; then
            logAndPrint "[WARNING]: ${short_install_dir} permission is too small, it is recommended that the permission be 755 for the root user."
            return 2
        elif [ ${mod_num} -eq 755 ]; then
            logAndPrint "[INFO]: ${short_install_dir} permission is ok."
        else
            logAndPrint "[WARNING]: ${short_install_dir} permission is too high, it is recommended that the permission be 755 for the root user."
            [ "${quiet_install}" = n ] && return 3
        fi
        parent_dirs_permision_check "${parent_dir}"
    fi
}

# 判断输入的指定路径是否存在
uninstallpathjudg() {
    if [ "$input_install_userpath" = "y" ];then
        getAbsolutePath "$input_install_path"
        if [ $? -eq 1 ];then
            logPrint "[INFuninstallpathjudgO]: uninstall path is invalid"
            log "[INFO]: uninstall path is invalid"
            exitLog 1
        fi
        input_install_path=${tmppath}
    fi
    if [ -d "$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel" ];then
        if [[ -L "$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scene.info"  || \
             -f "$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scene.info" ]];then
            logPrint "[INFO]: The old install path of opp_kernel:$input_install_path"
            installInfo="$input_install_path/opp_kernel/ascend_install.info"
            if [ -f "$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/script/run_opp_kernel_uninstall.sh" ]; then
                "$input_install_path"/opp/built-in/op_impl/ai_core/tbe/kernel/script/run_opp_kernel_uninstall.sh "$input_install_path" > /dev/null 2>&1
                if [ $? -eq 0 ];then
                    logPrint "[INFO]: remove kernel successfully"
                    log "[INFO]: remove kernel successfully"
                    logPrint "[INFO]: OPPKERNEL package uninstalled successfully! Uninstallation takes effect immediately."
                    log "[INFO]: OPPKERNEL package uninstalled successfully! Uninstallation takes effect immediately."
                    exitLog 0
                else
                    logPrint "[ERROR]: remove opp kernel failed"
                    log "[ERROR]: remove opp kernel failed"
                    exitLog 1
                fi
            fi
            if [ -f "$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/run_opp_kernel_uninstall.sh" ]; then
                "$input_install_path"/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/run_opp_kernel_uninstall.sh "$input_install_path" > /dev/null 2>&1
                if [ $? -eq 0 ];then
                    logPrint "[INFO]: remove kernel successfully"
                    log "[INFO]: remove kernel successfully"
                    logPrint "[INFO]: OPPKERNEL package uninstalled successfully! Uninstallation takes effect immediately."
                    log "[INFO]: OPPKERNEL package uninstalled successfully! Uninstallation takes effect immediately."
                    exitLog 0
                else
                    logPrint "[ERROR]: remove opp kernel failed"
                    log "[ERROR]: remove opp kernel failed"
                    exitLog 1
                fi
            fi
        else
            logPrint "[ERROR]: The old install path of opp kernel:$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel not exist"
            exitLog 1
        fi
    else
        logPrint "[ERROR]: The old install path of opp_kernel:$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel not exist"
        exitLog 1
    fi
}

runfile=$(expr substr "$2" 3 $(expr "${#2}" - 2))/$(expr substr "$1" 5 $(expr "${#1}" - 4))
runfilename=$(expr substr "$1" 5 $(expr "${#1}" - 4))
full_install=n
feature_install=n
uninstall=n
upgrade=n
opp_kernel_static=n
input_install_username=n
input_install_usergroup=n
input_install_userpath=n
old_install_path=""
input_install_path=""
installMode=""
quiet_install=n
hot_reset_support=n
#以下未实现
run_install=n
check=n
version=n

# cut first two params from *.run
i=0
while true
do
    if [ "x$1" = "x" ];then
        break
    fi
    if [ "`expr substr "$1" 1 2 `" = "--" ]; then
       i=`expr "$i" + 1`
    fi
    if [ $i -gt 2 ]; then
        break
    fi
    shift 1
done

start_time=$(date +"%Y-%m-%d %H:%M:%S")

all_parma="$@"

# isRoot
createLogFolder
changeLogMode
startLog

install_all=n
numop=0
numst=0
numother=0
onlycheck=n

if [ $(id -u) -eq 0 ]; then
    install_all=y
fi
while true
do
    case "$1" in
    -h | --help)
    paramUsage
    exitLog 0
    ;;
    #不支持在--uninstall后面传入地址参数
    --uninstall)
    uninstall=y
    let numop+=1
    let numst+=1
    shift
    ;;
    #不支持在--upgrade后面传入地址参数
    --upgrade)
    upgrade=y
    check=y
    installMode="full"
    let numop+=1
    shift
    ;;
    --full)
    full_install=y
    installMode="full"
    let numop+=1
    shift
    ;;
    --run)
    installMode="run"
    run_install=y
    check=y
    let numop+=1
    shift
    ;;
    --devel)
    full_install=y
    installMode="devel"
    check=y
    opp_kernel_static=y
    let numop+=1
    shift
    ;;
    --check)
    check=y
    onlycheck=y
    let numst+=1
    shift
    ;;
    --version)
    let numst+=1
    version=y
    shift
    ;;
    --quiet)
    let numother+=1
    quiet_install=y
    shift
    ;;
    --install-for-all)
    install_all=y
    shift
    ;;
    --feature=*)
        feature_choice=$(echo $1 | cut -d"=" -f2 )
        if test -z "$feature_choice"; then
            echo "[opp_kernel] [ERROR]: ERR_NO:0x0002;ERR_DES:Paramter --feature cannot be null."
            exitLog 1
        fi
        contain_feature "ret" "$feature_choice" "${_FILELIST_FILE}"
        if [ "$ret" = "false" ]; then
            logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:opp_kernel package doesn't contain features $feature_choice, skip installation."
            exit 1
        fi
        installMode="full"
        feature_install=y
        shift
        ;;
    --install-path=*)
    input_install_userpath=y
    temp_path=`echo "$1" | cut -d"=" -f2- `
    slashes_num=$( echo "${temp_path}" | grep -o '/' | wc -l )
    # 去除指定安装目录后所有的 "/"
    if [ $slashes_num -gt 1 ];then
        input_install_path=`echo "${temp_path}" | sed "s/\/*$//g"`
    else
        input_install_path="${temp_path}"
    fi
    shift
    ;;
    -*)
    echo "ERR_NO:0x0004;ERR_DES: Unsupported parameters : $1"
    paramUsage
    ;;
    *)
    break
    ;;
    esac
done

FeatureTypeCheck() {
    feature_list=$(echo $1 | awk '{split($0,arr,",");for(i in arr) print arr[i]}')
    local check_feature_list=("aclnn_ops_infer" "aclnn_ops_train" "aclnn_math" "aclnn_rand")
    for feature in ${feature_list[@]}; do
        if [[ ! "${check_feature_list[@]}" =~ "${feature}" ]]; then
            echo "[Opp_kernel]: ERR_NO:0x0004;ERR_DES: Parameter --feature"
            exitLog 1
        fi
    done
}

relationfeature() {
    local feature_all
    related_feature1="aclnn_ops_train" 
    related_feature2="aclnn_ops_infer"
    related_feature_base="aclnn_math"
    related_feature_rand="aclnn_rand"
    if [[ $1 =~ $related_feature1 ]];then
        feature_all="$1,$related_feature2,$related_feature_base,$related_feature_rand"
    else
        if [[  $1 =~ $related_feature2 ]];then
            feature_all="$1,$related_feature_base"
        else
            feature_all=$1
        fi
    fi
    # 去重
    feature_all=$(echo ${feature_all} | awk -F "," '{for(i=1;i<=NF;i++) if(!a[$i]) {a[$i]=1;printf("%s ", $i)}}' | tr ' ' ',')
    feature_all=${feature_all%%,}
    echo ${feature_all}
 }

in_feature_new=""
if [ -z "${feature_choice}" ] && [ "${full_install}" = "y" ] && [ "${installMode}" = "full" ]; then
    in_feature_new='all'
else
    FeatureTypeCheck ${feature_choice}
    if [ $? -eq 0 ];then
        in_feature_new=$(relationfeature ${feature_choice})
    fi
fi

checkInstallCondition
judgmentpath
is_multi_version_pkg "pkg_is_multi_version" "$_VERSION_INFO_FILE"
get_version_dir "pkg_version_dir" "$_VERSION_INFO_FILE"
origin_install_path="${input_install_path}"
if [ "$pkg_is_multi_version" = "true" ]; then
    input_install_path="${input_install_path}/$pkg_version_dir"
else
    input_install_path="${input_install_path}"
fi

if [ $numop -gt 1 ];then
    if [ $numst -gt 0 ];then
        logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
        log "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
        exitLog 1
    fi
    logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:only support one type: full/run/upgrade, operation failed!"
    log "[ERROR]: ERR_NO:0x0004;ERR_DES:only support one type: full/run/upgrade, operation failed!"
    exitLog 1
elif [ $numst -gt 1 ];then
    logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
    log "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
    exitLog 1
else
    if [ $numother -gt 0 ] && [ $numst = 1 ] && [ "x${uninstall}" != "xy" ];then
        logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
        log "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
        exitLog 1
    fi
    if [ $numop = 1 ] && [ $numst = 1 ] && [ "x${uninstall}" != "xy" ]; then
        logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
        log "[ERROR]: ERR_NO:0x0004;ERR_DES:version/uninstall/check can't use with other parameters."
        exitLog 1
    fi
fi

if [ $onlycheck = y ];then
    if [ -f ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/ver_check.sh ];then
        ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/ver_check.sh "${input_install_path}"
    fi
    if [ -f ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/script/ver_check.sh ];then
        ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/script/ver_check.sh "${input_install_path}"
    fi
    exitLog 0
fi

if [ $version = y ]; then
    logPrint "[INFO]: opp_kernels version :"$(getVersionInRunFile)
    exit 0
fi

# check parameters conflict
if [ "${docker_install}" = y ]; then
    # docker安装仅用于driver子包
    log "[ERROR]: ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
    logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
    logPrint "[ERROR]: --docker not used in opp kernel"
    exitLog 1
fi

if [ "${quiet_install}" = "y" ];then
    if [ "${upgrade}" = "y" ] || [ "${full_install}" = "y" ] || [ "${run_install}" = "y" ]  || [ "${uninstall}" = "y" ];then
        quiet_install=y
    else
        log "[ERROR]: ERR_NO:0x0004;ERR_DES:quiet param need used with full/run/uninstall/upgrade."
        logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:quiet param need used with full/run/uninstall/upgrade."
        exitLog 1
    fi
fi
if [ "${upgrade}" = "n" ] && [ "${full_install}" = "n" ] && [ "${run_install}" = "n" ]  && [ "${uninstall}" = "n" ];then
    log "[ERROR]: ERR_NO:0x0004;ERR_DES:param need used with full/run/uninstall/upgrade."
    logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:param need used with full/run/uninstall/upgrade."
    exitLog 1
fi
if [ "${upgrade}" = "y" ] && [ "${feature_install}" = "y" ];then
    log "[ERROR]: ERR_NO:0x0004;ERR_DES:feature param cannot used with upgrade."
    logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:feature param cannot used with upgrade."
    exitLog 1
fi
if [ $uninstall = y ]; then
    # 卸载参数只支持单独使用
    if [ "${upgrade}" = y ] || [ "$full_install" = y ] || [ "$run_install" = y ] || [ "$feature_install" = y ]; then
        log "ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
        logPrint "[ERROR]: ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
        exitLog 1
    fi
    uninstallpathjudg
fi

if [ "${upgrade}" = y ];then
    if [ "$full_install" = n ] || [ "$run_install" = n ];then
        get_package_upgrade_install_info "installInfo" "$origin_install_path" "opp_kernel"
        if [ -f ${installInfo} ];then
            Opp_Kernels_Install_Type=$(getInstallParam "OPP_KERNEL_INSTALL_TYPE" "${installInfo}")
            installMode=$Opp_Kernels_Install_Type
        else
            if [ -f $input_install_path/opp_kernel/scene.info ];then
                touch $installInfo
                if [ $(id -u) -eq 0 ];then
                    chown -RPf "${username}:$usergroup" "$installInfo"
                fi
                chmod 600 "$installInfo" > /dev/null 2>&1
                updateInstallParam "USERNAME" "$username" "$installInfo"
                updateInstallParam "USERGROUP" "$usergroup" "$installInfo"
                updateInstallParam "OPP_KERNEL_INSTALL_TYPE" "$installMode" "$installInfo"
                updateInstallParam "OPP_KERNEL_FEATURE_TYPE" "$in_feature_new" "$installInfo"
                updateInstallParam "OPP_KERNEL_INSTALL_PATH_PARAM" "$input_install_path" "$installInfo"
            else
                if [ ! -d $input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/ ];then
                    logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, upgrade failed"
                    log "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, upgrade failed"
                    exitLog 1
                fi
            fi
        fi
    fi
fi

get_arch "arch_info" "$_SCENE_INFO_FILE"

# check platform
if [ "${architecture}" != "${arch_info}" ] && [ "${arch_info}" != "UNKNOWN" ] ; then  
    logPrint "[ERROR]: ERR_NO:0x0001;ERR_DES:the architecture of the run package is inconsistent with that of the current environment. "
    log "[ERROR]: ERR_NO:0x0001;ERR_DES:the architecture of the run package is inconsistent with that of the current environment. "
    exitLog 1
fi

if [ "$opp_kernel_static" = "y" ]; then
    # 生成opp_kernel_static目录
    path_in_pkg="opp_kernel_static"
    rm -rf ${path_in_pkg}
    mkdir -p ${path_in_pkg}

    # 解压xz软件
    tar -Jxf ./Ascend910B-opp_kernel_static-*.tar.xz -C ${path_in_pkg}

    # 后续卸载操作依赖filelist, 使用接口写入filelist文件， 依赖get_arch接口获取架构
    files_cmd=$(ls ${path_in_pkg}/lib64)
    for file_name in $files_cmd
    do
        install_path="${arch_info}-linux/lib64/${file_name}"

        # 使用 add_fileitem 将文件路径写入文件
        add_fileitem "$_FILELIST_FILE" "copy" "$path_in_pkg/lib64/${file_name}" "$install_path" "440" "DEFAULT" "devel" \
                     "NA" "opp/lib64/${file_name}" "opp_kernel" "DEFAULT" "DEFAULT" "$arch_info"
    done

fi

# 是否跨产品形态
cross_product() {
    local _outvar="$1"
    local _result

    if [ ! -f "$input_install_path/opp_kernel/ascend_install.info" ]; then
        eval "${_outvar}=\"none\""
    else
        eval "${_outvar}=\"false\""
    fi
}

cross_product "is_cross_product"

isValidPath
getOldAndInputPath

getUserInfo

if [ $check = y ];then
    if [ -f ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/ver_check.sh ];then
        ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/ver_check.sh "${input_install_path}"
    fi
    if [ -f ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/script/ver_check.sh ];then
        ${input_install_path}/opp/built-in/op_impl/ai_core/tbe/kernel/script/ver_check.sh "${input_install_path}"
    fi
fi

if [ $? -ne 0 ];then
    logPrint "[INFO]: do you want to continue installing? [y/n]"
    if [ $quiet_install = n ];then
        while true
        do
            read yn
            if [ "$yn" = n ]; then
                logPrint "[INFO]: stop installation!"
                exitLog 0;
            elif [ "$yn" = y ]; then
                break;
            else
                logPrint "[ERROR]: ERR_NO:0x0002;ERR_DES:input error, please input again!"
            fi
        done
    fi
fi

stash_binary_configs() {
    local base_dir="$1"
    local product="$2"
    local mod_script mod_ascend910b mod_json
    shift 2

    chmod u+w "$base_dir/script"
    chmod u+w "$base_dir/config/ascend910b"

    cp -f "$base_dir/script/filelist.csv" "$base_dir/script/filelist.csv.stash"
    cp -f "$base_dir/config/ascend910b/binary_info_config.json" "$base_dir/config/ascend910b/binary_info_config.json.stash"

    "$@"

    mod_script="$(stat -L -c %a "$base_dir/script")"
    mod_ascend910b="$(stat -L -c %a "$base_dir/config/ascend910b")"
    mod_json="$(stat -L -c %a "$base_dir/config/ascend910b/binary_info_config.json")"

    chmod u+w "$base_dir/script"
    chmod u+w "$base_dir/config/ascend910b"
    chmod u+w "$base_dir/config/ascend910b/binary_info_config.json"

    python3 $base_dir/script/merge_binary_info_config.py --base-file $base_dir/config/ascend910b/binary_info_config.json.stash --update-file $base_dir/config/ascend910b/binary_info_config.json --output-file $base_dir/config/ascend910b/binary_info_config.json
    rm -f "$base_dir/config/ascend910b/binary_info_config.json.stash"
    mv -f "$base_dir/script/filelist.csv.stash" "$base_dir/script/filelist.csv"

    chmod "$mod_json" "$base_dir/config/ascend910b/binary_info_config.json"
    chmod "$mod_ascend910b" "$base_dir/config/ascend910b"
    chmod "$mod_script" "$base_dir/script"
}

mergeFile() {
    local base_dir="$1"
    shift 1

    # replace the symbolic variable in filelist.csv
    newStr="${architectureDir}"
    oldStr="\$(TARGET_ENV)"
    sed -i 's#'''$oldStr'''#'''$newStr'''#g' ${_FILELIST_FILE}

    cat "$base_dir/script/filelist.csv" ${_FILELIST_FILE} | awk -F ',' '!seen[$4]++' > ./merge_file.csv
    "$@"
    mod_script="$(stat -L -c %a "$base_dir/script")"
    mod_filelist="$(stat -L -c %a "$base_dir/script/filelist.csv")"
    chmod u+w "$base_dir/script"
    chmod u+w "$base_dir/script/filelist.csv"
    mv -f ./merge_file.csv "$base_dir/script/filelist.csv"
    chmod "$mod_filelist" "$base_dir/script/filelist.csv"
    chmod "$mod_script" "$base_dir/script"
}

if echo "$runfilename" | grep "\-fusion_kernel" > /dev/null 2>&1; then
    is_fusion_kernel="true"
else
    is_fusion_kernel="false"
fi

if [ "$is_fusion_kernel" = "true" ]; then
    # 卸载场景
    if [ $uninstall = y ]; then
        if [ "$is_cross_product" = "none" ]; then
            logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, uninstall failed"
            log "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, uninstall failed"
            exitLog 1
        else
            uninstallRun "uninstall"
        fi
    # 安装场景
    elif [ $full_install = y ] || [ $run_install = y ]; then
        if [ "$is_cross_product" = "false" ]; then
            stash_func="stash_binary_configs \"$input_install_path/opp/built-in/op_impl/ai_core/tbe/kernel\" ascend910b"
        else
            stash_func=""
        fi
        eval $stash_func installRun "install" $quiet_install $hot_reset_support
    elif [ $upgrade = y ];then
        logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, upgrade failed"
        log "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, upgrade failed"
        exitLog 1
    fi
else
    # 环境上是否已安装过run包
    if [ ! -z "$old_install_path" ]; then
        version_res_cur=($(getVersionInRunFile))
        version1_cur=${version_res_cur[0]}
        version2_cur=${version_res_cur[1]}
        if [ -d "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" ] || [ -d "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel" ];then
            version_res=($(getVersionInstalled "$old_install_path/opp_kernel"))
            version1_old=${version_res[0]}
            version2_old=${version_res[1]}
        fi
        # 卸载场景
        if [ $uninstall = y ]; then
            uninstallRun "uninstall"
        # 升级或者覆盖安装场景
        elif [ $full_install = y ] || [ $upgrade = y ] || [ $run_install = y ]; then
            if [ ! $upgrade = y ]; then
                # 增量安装
                if [ $feature_install = y ]; then
                    runpkg_version=""
                    if [ -f "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scene.info" ];then
                        . "$old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scene.info"
                        runpkg_version=${version}
                    else
                        log "[INFO]: runpkg not installed"
                        exitLog 0
                    fi
                    # run 版本校验
                    if [ ! $runpkg_version"x" = "x" ];then
                        if [ $version1_cur"x" = $runpkg_version"x" ]; then
                            log "[INFO]: Opp_kernel package has been installed on the path ${old_install_path}, the version is ${version1_cur}, and the version of this run package is ${runpkg_version}."
                            old_feature=$(getInstallParam "OPP_KERNEL_FEATURE_TYPE" "${installInfo}")
                            if [ "${old_feature}" = "all" ]; then
                                in_feature_all=${old_feature}
                            elif [ "${old_feature}" != "" ] && [ ! "${old_feature}" = "all" ]; then
                                in_feature_all="${old_feature},${in_feature_new}"
                                in_feature_all=$(echo ${in_feature_all} | awk -F "," '{for(i=1;i<=NF;i++) if(!a[$i]) {a[$i]=1;printf("%s ", $i)}}' | tr ' ' ',') # 去重
                                in_feature_all=${in_feature_all%%,}
                            else
                                in_feature_all=${in_feature_new}
                            fi
                            stash_func="mergeFile \"$old_install_path/opp_kernel\""
                            eval $stash_func installRun "install" $quiet_install $hot_reset_support
                        else
                            log "[INFO]: Opp_kernel package not match."
                            exitLog 1
                        fi
                    else
                        old_feature=$(getInstallParam "OPP_KERNEL_FEATURE_TYPE" "${installInfo}")
                        if [ "${old_feature}" = "all" ]; then
                            in_feature_all=${old_feature}
                        elif [ "${old_feature}" != "" ] && [ ! "${old_feature}" = "all" ]; then
                            in_feature_all="${old_feature},${in_feature_new}"
                            in_feature_all=$(echo ${in_feature_all} | awk -F "," '{for(i=1;i<=NF;i++) if(!a[$i]) {a[$i]=1;printf("%s ", $i)}}' | tr ' ' ',') # 去重
                            in_feature_all=${in_feature_all%%,}
                        else
                            in_feature_all=${in_feature_new}
                        fi
                        stash_func="mergeFile \"$old_install_path/opp_kernel\""
                        eval $stash_func installRun "install" $quiet_install $hot_reset_support
                    fi
                else
                    if [ ! $version2_cur"x" = "x" ] && [ ! "$version2_cur" = "none" ]; then
                        # 判断是否要覆盖式安装
                        logPrint "[INFO]: Opp_kernel package has been installed on the path ${old_install_path}, the version is ${version2_old}, and the version of this package is ${version2_cur}, do you want to continue? [y/n]"
                        if [ $quiet_install = n ];then
                            while true
                            do
                                read yn
                                if [ "$yn" = n ]; then
                                    logPrint "[INFO]: stop installation!"
                                    exitLog 0;
                                elif [ "$yn" = y ]; then
                                    break;
                                else
                                    logPrint "[ERROR]: ERR_NO:0x0002;ERR_DES:input error, please input again!"
                                fi
                            done
                        fi
                    fi
                    colorPrint "[WARNING]:\033[33m Old run package exists, and new run package will be installed \033[0m"
                    log "[WARNING]: Old run package exists, and new run package will be installed"
                    in_feature_all=${in_feature_new}
                    stash_func="mergeFile \"$old_install_path/opp_kernel\""
                    eval $stash_func installRun "install" $quiet_install $hot_reset_support
                fi
            else
                if [ -f $old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/scripts/uninstall.sh ] || [ -f $old_install_path/opp/built-in/op_impl/ai_core/tbe/kernel/script/uninstall.sh ];then
                    uninstallRun "upgrade"
                fi
                installRun "upgrade" $quiet_install $hot_reset_support
            fi
        fi
    else
        # 卸载场景
        if [ $uninstall = y ]; then
            logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, uninstall failed"
            log "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, uninstall failed"
            exitLog 1
            # logOperation "${LOG_OPERATION_UNINSTALL}" "${start_time}" "${runfilename}" "${LOG_RESULT_FAILED}" "${installMode}" "${all_parma}"
        # 安装场景
        elif [ $full_install = y ] || [ $run_install = y ] || [ $feature_install = y ]; then
            in_feature_all=$in_feature_new
            installRun "install" $quiet_install $hot_reset_support
        elif [ $upgrade = y ];then
            logPrint "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, upgrade failed"
            log "[ERROR]: ERR_NO:0x0080;ERR_DES:Runfile is not installed on this device, upgrade failed"
            exitLog 1
            # logOperation "${LOG_OPERATION_UNINSTALL}" "${start_time}" "${runfilename}" "${LOG_RESULT_FAILED}" "${installMode}" "${all_parma}"
        fi
    fi
fi

writeSource $input_install_path
exitLog 0
