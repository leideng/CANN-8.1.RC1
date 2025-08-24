#!/bin/bash
# Perform install/upgrade/uninstall for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

default_root_dir="/usr/local/Ascend"
default_normal_dir="${HOME}/Ascend"
default_normal_username=$(id -nu)
default_narmal_usergroup=$(id -ng)
username=HwHiAiUser
usergroup=HwHiAiUser
curpath=$(dirname $(readlink -f $0))
common_func_path="${curpath}/common_func.inc"
common_func_v2_path="${curpath}/common_func_v2.inc"
version_cfg_path="${curpath}/version_cfg.inc"
pkg_version_path="${curpath}/../../version.info"
pkg_info_file="${curpath}/../scene.info"
platform_data=$(grep -e "arch" "$pkg_info_file" | cut --only-delimited -d"=" -f2-)

install_info_old="/etc/ascend_install.info"
RUN_DIR="$(echo $2 | cut -d'-' -f 3)"

# load shell
. "${common_func_path}"
. "${common_func_v2_path}"
. "${version_cfg_path}"
COMMON_SHELL="${curpath}/common.sh"
source "${COMMON_SHELL}"

# modify user and path
if [[ "$(id -u)" != "0" ]]; then
    username="${default_normal_username}"
    usergroup="${default_narmal_usergroup}"
    default_root_dir=${default_normal_dir}
fi

# 递归授权
chmod_recur() {
    local file_path="${1}"
    local permision="${2}"
    local type="${3}"
    local permission=$(set_file_chmod $permision)
    if [ "$type" = "dir" ]; then
        find "$file_path" -type d -exec chmod $permission {} \; 2> /dev/null
    elif [ "$type" = "file" ]; then
        find "$file_path" -type f -exec chmod $permission {} \; 2> /dev/null
    fi
}

#单目录授权
chmod_single_dir() {
    local file_path="${1}"
    local permision="${2}"
    local type="${3}"
    local permission=$(set_file_chmod $permision)
    if [ "$type" = "dir" ]; then
        chmod $permission "$file_path"
    elif [ "$type" = "file" ]; then
        chmod $permission "$file_path"
    fi
}

#设置权限
set_file_chmod() {
    local permission="${1}"
    local new_permission
    if [[ "${input_install_for_all}" = "y" ]]; then
        new_permission="$(expr substr $permission 1 2)$(expr substr $permission 2 1)"
        echo $new_permission
    else
        echo $permission
    fi
}

# 运行前授权
chmod_start() {
    chmod_recur $default_dir 750 dir
}

# 运行结束授权
chmod_end() {
    # other file permission
    chmod_single_dir "${default_dir}"/ascend_install.info 600 file 2> /dev/null
    chmod_single_dir "${default_dir}" 550 file 2> /dev/null
    chmod_recur "${default_dir}/script" 500 file 2> /dev/null
    chmod_single_dir "${default_dir}/script" 500 dir 2> /dev/null
    chown -R "$username":"$usergroup" "${default_dir}" 2> /dev/null
    if [ $(id -u) -eq 0 ]; then
        chown "root:root" "$default_dir" 2> /dev/null
        chmod 755 "$default_dir" 2> /dev/null
        chown -R "root:root" "${default_dir}/script" 2> /dev/null
    fi
}

param_usage() {
    log "INFO" "Please input this command for help: ./${runfilename} --help"
    #cat ${usageFile}
}

# 日志文件轮询、备份
rotate_log() {
    echo "${log_file} {
        su root root
        daily
        size=5M
        rotate 3
        missingok
        create 440 root root
    }" > /etc/logrotate.d/HiAI-install
}

# 修改日志文件的权限
change_log_mode() {
    local _log_file=${1}
    if [ ! -f "$_log_file" ]; then
        touch $_log_file
    fi
    chmod_single_dir $_log_file 640 file
}

# 创建文件夹
create_folder() {
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
    if [ $(id -u) -ne 0 ]; then
        chmod_single_dir $l 740 dir
    else
        chmod_single_dir $1 750 dir
    fi
}

new_echo() {
    local log_type_=${1}
    local log_msg_=${2}
    if  [ "${is_quiet}" = "n" ]; then
        echo ${log_type_} ${log_msg_} 1 > /dev/null
    fi
}


# 开始安装前打印开始信息
start_install_log() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    new_echo "INFO" "Start time:${cur_date}"
    log "INFO" "Start time:$cur_date"
    log "INFO" "LogFile:${log_file}"
    log "INFO" "InputParams:$all_parma"
    log "INFO" "OperationLogFile:${operation_logfile}"
}
# 开始卸载前打印开始信息
start_uninstall_log() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    new_echo "INFO" "Start time:${cur_date}"
    log "INFO" "Start time:$cur_date"
    log "INFO" "LogFile:${log_file}"
    log "INFO" "InputParams:$all_parma"
    log "INFO" "OperationLogFile:${operation_logfile}"
}

# 安装结束退出前打印结束信息
exit_install_log() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    new_echo "INFO" "End time:${cur_date}"
    log "INFO" "End time:${cur_date}"
    exit $1
}

# 安装结束退出前打印结束信息
exit_uninstall_log() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    new_echo "INFO" "End time:${cur_date}"
    log "INFO" "End time:${cur_date}"
    exit $1
}

# 安全日志
log_operation(){
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    if [ "$1"  = "Install" ]; then
        level="SUGGESTION"
    elif [ "$1" = "Upgrade" ]; then
        level="MINOR"
    elif  [ "$1" = "Uninstall" ]; then
        level="MAJOR"
    else
        level="UNKNOWN"
    fi

    if [ ! -f "${operation_logfile}" ]; then
        touch ${operation_logfile}
        chmod_single_dir ${operation_logfile} 640 file
    fi

    echo "$1 $level root $cur_date 127.0.0.1 $runfilename $2 install_mode=$install_mode; cmdlist=$all_parma" >> $operation_logfile
}

#相对路径转化绝对路径
relative_path_to_absolute_path() {
    local relative_path_="${1}"
    local fstr="$(expr substr "$relative_path_" 1 1)"
    if [ "$fstr" = "./" ];then
        local absolution_path_="${RUN_DIR}/${relative_path_}"
        echo "${absolution_path_}"
    else
        echo "${1}"
    fi
}

get_dir_mod() {
    local path="$1"
    stat -c %a "$path"
}

remove_dir_recursive() {
    local dir_start="$1"
    local dir_end="$2"
    if [ "$dir_end" = "$dir_start" ]; then
        return 0
    fi
    if [ ! -e "$dir_end" ]; then
        return 0
    fi
    if [ "x$(ls -A $dir_end 2>&1)" != "x" ]; then
        return 0
    fi
    local up_dir="$(dirname $dir_end)"
    local oldmod="$(get_dir_mod $up_dir)"
    chmod u+w "$up_dir"
    [ -n "$dir_end" ] && rm -rf "$dir_end"
    if [ $? -ne 0 ]; then
        chmod "$oldmod" "$up_dir"
        return 1
    fi
    chmod "$oldmod" "$up_dir"
    remove_dir_recursive "$dir_start" "$up_dir"
}

# get install path
get_install_path() {
    local temp_path_install
    if [ "$input_path_flag" = y ]; then
        if [ x"${input_install_path}" = "x" ]; then
            log "ERROR" "ERR_NO:0x0004;ERR_DES: Install path is empty."
            exit_log 1
        fi
        # delete last "/"
        temp_path_install="${input_install_path}"
        temp_path_install=$(echo "${temp_path_install}" | sed "s/\/*$//g")
        if [ x"${temp_path_install}" = "x" ]; then
            temp_path_install="/"
        fi
        prefix=$(echo "${temp_path_install}" | cut -d"/" -f1 | cut -d"~" -f1)
        if [ x"${prefix}" = "x" ]; then
            install_dir="${temp_path_install}"
        else
            prefix=$(echo "${run_path}" | cut -d"/" -f1 | cut -d"~" -f1)
            if [ x"${prefix}" = "x" ]; then
                install_dir="${run_path}/${temp_path_install}"
            else
                log "ERROR" "ERR_NO:0x0004;ERR_DES: Run package path is invalid: $run_path"
                exit_log 1
            fi
        fi
        home=$(echo "${install_dir}" | cut -d"~" -f1)
        if [ "x${home}" = "x" ]; then
            temp_path_install=$(echo "${install_dir}" | cut -d"~" -f2)
            if [ "$(id -u)" -eq 0 ]; then
                install_dir="/root${temp_path_install}"
            else
                install_dir="/home/$(whoami)${temp_path_install}"
            fi
        fi
    else
        # use default path
        if [ $? -eq 0 ]; then
            install_dir=$default_root_dir
        else
            install_dir="/home/$(whoami)/Ascend"
        fi
    fi
    if [ ! "$uninstall" = y ]; then
        # check dir is exist
        ppath=$(echo "${install_dir}" | sed "s/\/[^/]*$//g")
        if [ ! "x${ppath}" = "x" ] && [ ! -d "${ppath}" ]; then
            log "ERROR" "parent path doesn't exist, please create ${ppath} first."
            exit_log 1
        fi
    fi

    echo "${install_dir}"
}

create_install_dir() {
    local path=$1
    local user_and_group=$2
    local permision

    if [ $(id -u) -eq 0 ]; then
        user_and_group="root:root"
        permission=755
    else
        if [ "$input_install_for_all" = y ]; then
            permission=755
        else
            permission=750
        fi
    fi

    if [ x"${path}" = "x" ]; then
        log "WARNING" "dir path is empty"
        return 1
    fi
    mkdir -p "${path}"
    if [ $? -ne 0 ]; then
        log "WARNING" "create path="${path}" failed."
        return 1
    fi
    chmod -f $permission "${path}"
    if [ $? -ne 0 ]; then
        log "WARNING" "chmod path="${path}" $permission failed."
        return 1
    fi
    chown -f $user_and_group "${path}"
    if [ $? -ne 0 ]; then
        log "WARNING" "chown path="${path}" $user_and_group failed."
        return 1
    fi
}

create_file() {
    local _file=$1
    if [ -d "${_file}" ]; then
        test -f ${_file} && rm -rf "${_file}"
    fi
    touch "${_file}"
    chown "$2" "${_file}"
    chmod_single_dir "${_file}" "$3" file
    if [ $? -ne 0 ]; then
        return 1
    fi
    return 0
}

# 判断输入的指定路径是否存在
is_valid_path() {
    if [ ! "${install_path}" = "" ]; then
        if [ ! -d "${install_path}" ]; then
            local up_dir=$(dirname "${install_path}")
            if [ "${install_path}" = "${default_root_dir}" ] || [ "${install_path}" = "${default_normal_dir}" ] || [ -d "${up_dir}" ];then
                create_install_dir "${install_path}" "${username}":"${usergroup}"
                install_path=$(cd "${input_install_path}";pwd)
            else
                # 路径不存在，报错
                log "ERROR" "ERR_NO:0x0003;ERR_DES:The $install_path dose not exist, please retry a right path."
                exit_install_log 1
            fi
        else
            local ret=0
            if [ $(id -u) -eq 0 ]; then
                parent_dirs_permision_check "${install_path}" && ret=$? || ret=$?
                if [ "${is_quiet}" = y ] && [ "${ret}" -ne 0 ]; then
                    log "ERROR" "the given dir, or its parents, permission is invalid."
                    exit 1
                fi
                if [ ${ret} -ne 0 ]; then
                    log "WARNING" "You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
                    while true
                    do
                        read yn
                        if [ "$yn" = n ]; then
                            exit 1
                        elif [ "$yn" = y ]; then
                            break;
                        else
                            echo "ERR_NO:0x0002;ERR_DES:input error, please input again!"
                        fi
                    done
                fi
            fi
            install_path=$(cd "${install_path}";pwd)
            if [ $(id -u) -ne 0 ]; then
                #cd $input_install_path >> /dev/null 2>&1
                log "DEBUG" "$install_path"
            else
                sh -c "cd \"$install_path\" > /dev/null 2>&1"
            fi
            if [ ! $? = 0 ]; then
                log "ERROR" "ERR_NO:0x0093;ERR_DES:The $username do not have the permission to access $install_path, please reset the directory to a right permission."
                exit_install_log 1
            fi
        fi
    fi
}

parent_dirs_permision_check() {
    local current_dir="$1"
    local parent_dir=""
    local short_install_dir=""
    local owner=""
    local mod_num=""

    parent_dir=$(dirname "${current_dir}")
    short_install_dir=$(basename "${current_dir}")
    log "INFO" "parent_dir value is [${parent_dir}] and children_dir value is [${short_install_dir}]"

    if [ "${current_dir}"x = "/"x ]; then
        log "INFO" "parent_dirs_permision_check success"
        return 0
    else
        owner=$(stat -c %U "${parent_dir}"/"${short_install_dir}")
        if [ "${owner}" != "root" ]; then
            log "WARNING" "[${short_install_dir}] permision isn't right, it should belong to root."
            return 1
        fi
        log "INFO" "[${short_install_dir}] belongs to root."

        mod_num=$(stat -c %a "${parent_dir}"/"${short_install_dir}")
        mod_num=$(check_chmod_length $mod_num)
        if [ ${mod_num} -lt 755 ]; then
            log "WARNING" "[${short_install_dir}] permission is too small, it is recommended that the permission be 755 for the root user."
            return 2
        elif [ ${mod_num} -eq 755 ]; then
            log "INFO" "[${short_install_dir}] permission is ok."
        else
            log "WARNING" "[${short_install_dir}] permission is too high, it is recommended that the permission be 755 for the root user."
            [ "${is_quiet}" = n ] && return 3
        fi
        parent_dirs_permision_check "${parent_dir}"
    fi
}

check_chmod_length() {
    local mod_num=$1
    local new_mod_num
    mod_num_length=$(expr length "$mod_num")
    if [ $mod_num_length -eq 3 ]; then
        new_mod_num=$mod_num
        echo $new_mod_num
    elif [ $mod_num_length -eq 4 ]; then
        new_mod_num="$(expr substr $mod_num 2 3)"
        echo $new_mod_num
    fi
}

# 检查是否存在进程
check_process() {
    name=$(ps -ef | awk '$2=='$$'{print $10}'|rev|cut -d "/" -f1|rev)
    shellname=$(echo $0 |rev |cut -d "/" -f1 |rev)
    process=$(ps -ef | grep -v "grep" | grep -w "$shellname" |grep -w "$name")
    pid=$(echo "$process" | awk -F ' ' '{print $2}')
    ret=$(echo "$process" | awk -F ' ' '{print $3}' | grep -v "$pid" | wc -l)
    if [ $ret -gt 1 ]; then
        log "ERROR" "ERR_NO:0x0094;ERR_DES:There is already a process being executed,please do not execute multiple tasks at the same time"
        log "DEBUG" "$name;$shellname;$ret;$process"
        exit_install_log 1
    fi
}

# 校验user和group的关联关系
check_group() {
    result=$(groups "$2" | grep ":")
    if [[ "${result}X" != "X" ]]; then
        group_user_related=$(groups "$2"|awk -F":" '{print $2}'|grep -w "$1")
    else
        group_user_related=$(groups "$2"|grep -w "$1")
    fi
    if [ "${group_user_related}x" != "x" ];then
        return 0
    else
        return 1
    fi
}

#安装路径加锁
chattr_files(){
    if [ -f "$install_info" ]; then
        #安装路径存在时加锁
        if [ -d "${ncs_install_path_param}/${pkg_relative_path}" ]; then
            chattr -R +i "${ncs_install_path_param}/${pkg_relative_path}" >> /dev/null 2>&1
            if [ $? -ne 0 ];then
                log "DEBUG" "chattr +i for the subfiles."
                find ${ncs_install_path_param}/${pkg_relative_path} -name "*" | xargs chattr +i  >> /dev/null 2>&1
            else
                log "DEBUG" "chattr -R +i $ncs_install_path_param succeeded."
            fi
        fi
    fi
}

#解锁
unchattr_files() {
    local unchattr_dir=${1}
    if [ -f "${install_info}" ]; then
        if [ -d "${unchattr_dir}" ]; then
            chattr -R -i "${unchattr_dir}" >> /dev/null 2>&1
            if [ $? -ne 0 ];then
                log "DEBUG" "unchattr -i for the subfiles."
                find ${unchattr_dir} -name "*" | xargs chattr -i  >> /dev/null 2>&1
            else
                log "DEBUG" "unchattr -R -i $unchattr_dir succeeded."
            fi
        fi
    fi
}

get_user_name() {
    local temp_user_name="${username}"
    if [ -f "${install_info}" ]; then
        local ncs_user_name=$(get_install_param "ncs_user_name" "${install_info}")
        if [ "x${ncs_user_name}" != "x" ]; then
            temp_user_name=${ncs_user_name}
        fi
    fi
    echo "${temp_user_name}"
}

get_user_group() {
    local temp_user_group="${usergroup}"
    if [ -f "${install_info}" ]; then
        local ncs_user_group=$(get_install_param "ncs_user_group" "${install_info}")
        if [ "x${ncs_user_group}" != "x" ]; then
            temp_user_group=${ncs_user_group}
        fi
    fi
    echo "${temp_user_group}"
}

#创建普通用户的默认安装目录
create_default_install_dir_for_commonuer() {
    if [ $(id -u) -ne 0 ]; then
        if [ "$input_path_flag" = n ]; then
            if [ ! -d "$install_path" ]; then
                create_install_dir "${install_path}" "${username}:${usergroup}"
            fi
        fi
    fi
}

# 获取安装目录下的完整版本号  version2
get_version_installed() {
    local version2=${version}
    if [ -f "$1"/version.info ]; then
        version2="$(grep -iw Version "$1/version.info" | cut -d"=" -f2-)"
    fi
    echo "$version2"
}

# 获取run包中的完整版本号  version1
get_version_run_file() {
    local version1="none"
    if [ -f "${pkg_version_path}" ]; then
        version1="$(grep -iw Version ${pkg_version_path} | cut -d"=" -f2-)"
    fi
    echo "$version1"
}

log_base_version() {
    if [ -f "${default_dir}" ];then
        installed_version=$(get_version_installed "${default_dir}")
        if [ ! "${installed_version}"x = ""x ]; then
            log "INFO" "base version is ${installed_version}."
            return 0
        fi
    fi
    log "WARNING" "base version was destroyed or not exist."
}

update_install_path() {
    if [ ! -d "${install_path}" ]; then
        # 路径不存在，报错
        log "ERROR" "ERR_NO:0x0003;ERR_DES:The $install_path dose not exist, please retry a right path."
        exit_install_log 1
    fi
}

update_install_param() {
    local _key=$1
    local _val="$2"
    local _file="$3"
    local install_info_key_array=("ncs_install_type" "ncs_user_name" "ncs_user_group" "ncs_install_path_param")
    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            local _param=$(grep -r "${_key}=" "${_file}")
            if [ "x${_param}" = "x" ]; then
                echo "${_key}=${_val}" >> "${_file}"
            else
                sed -i "/^${_key}=/c ${_key}=${_val}" "${_file}"
            fi
            break
        fi
    done
}

update_install_info() {
    local ncs_input_install_path=$1
    if [ -f "$install_info" ]; then
        local ncs_install_type=$(get_install_param "ncs_install_type" "${install_info}")
        local ncs_user_name=$(get_install_param "ncs_user_name" "${install_info}")
        local ncs_user_group=$(get_install_param "ncs_user_group" "${install_info}")
        local ncs_input_install_path=$(get_install_param "ncs_install_path_param" "${install_info}")
        if [[ "${is_docker_install}" == y ]] ; then
            local ncs_install_path_param=$(concat_docker_install_path "${docker_root}" "${ncs_input_install_path}")
        else
            local ncs_install_path_param=${ncs_input_install_path}
        fi
        if [ "$pkg_is_multi_version" = "true" ]; then
            ncs_install_path_param="${ncs_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
        else
            ncs_install_path_param="${ncs_install_path_param}/${pkg_relative_path}"
        fi
        if [ -z "$ncs_install_type" ]; then
            update_install_param "ncs_install_type" "${install_mode}" "${install_info}"
        else
            chmod_single_dir "${ncs_install_path_param}" 755 dir
            update_install_param "ncs_install_type" "${install_mode}" "${install_info}"
        fi

        if [ -z "${ncs_user_name}" ]; then
            update_install_param "ncs_user_name" "${username}" "${install_info}"
        fi

        if [ -z "${ncs_user_group}" ]; then
            update_install_param "ncs_user_group" "${usergroup}" "${install_info}"
        fi

        if [ -z "${ncs_install_path_param}" ]; then
            update_install_param "ncs_install_path_param" "${install_path_param}" "${install_info}"
        else
            chmod_single_dir "${ncs_install_path_param}" 755 dir
            update_install_param "ncs_install_path_param" "${install_path_param}" "${install_info}"
        fi
    else
        create_file "$install_info" "${username}":"${usergroup}" 600
        update_install_param "ncs_install_type" "${install_mode}" "${install_info}"
        update_install_param "ncs_user_name" "${username}" "${install_info}"
        update_install_param "ncs_user_group" "${usergroup}" "${install_info}"
        update_install_param "ncs_install_path_param" "${install_path_param}" "${install_info}"
    fi
}

set_env_var() {
    local pkg_install_path="${1}"
    local install_type=${2}
    if [ "${install_type}" = "full" ] || [ "${install_type}" = "run" ] || [ "${install_type}" = "upgrade" ]; then
    echo "Please make sure that
          - LD_LIBRARY_PATH includes ${pkg_install_path}/lib64"
    fi
}

check_docker_path(){
    local docker_path="$1"
    if [[ "${docker_path}" != "/"* ]]; then
        log "ERROR" "ERR_NO:0x0002;ERR_DES:Parameter --docker-root \
        must with absolute path that which is start with root directory /. Such as --docker-root=/${docker_path}"
        exit_install_log 1
    fi
    if [[ ! -d "${docker_path}" ]]; then
        log "ERROR" "ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist, please create this directory."
        exit_install_log 1
    fi
}

concat_docker_install_path() {
    local docker_path="$1"
    local input_install_path="$2"
    # delete last "/"
    docker_path=$(echo "${docker_path}" | sed "s/\/*$//g")
    if [ x"${docker_path}" = "x" ]; then
        docker_path="/"
    fi
    local install_path=${docker_path}${input_install_path}
    echo "${install_path}"
}

# 安装调用子脚本
install_run() {
    local operation="Install"
    update_install_path
    update_install_info "${ncs_input_install_path}"
    ncs_input_install_path=$(get_install_param "ncs_install_path_param" "${install_info}")
    local ncs_install_type=$(get_install_param "ncs_install_type" "${install_info}")
    if [[ "${is_docker_install}" == y ]] ; then
        local ncs_install_path_param=$(concat_docker_install_path "${docker_root}" "${ncs_input_install_path}")
    else
        local ncs_install_path_param=${ncs_input_install_path}
    fi
    if [ "$pkg_is_multi_version" = "true" ]; then
        ncs_install_path_param="${ncs_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
    else
        ncs_install_path_param="${ncs_install_path_param}/${pkg_relative_path}"
    fi
    unchattr_files $ncs_install_path_param
    if [ "$1" = "install" ]; then
        chmod_start
        new_echo "INFO" "install ${ncs_install_path_param} ${ncs_install_type}"
        log "INFO" "install ${ncs_install_path_param} ${ncs_install_type}"
        bash ${curpath}/run_ncs_install.sh "install" "$ncs_input_install_path" $ncs_install_type "$is_quiet" \
            "$input_setenv" "${is_docker_install}" "$docker_root" "$in_install_for_all"
        if [ $? -eq 0 ]; then
            log "INFO" "Ncs package installed successfully! The new version takes effect immediately."
            log_operation "${operation}" "success"
            chmod_end
            set_env_var "${ncs_install_path_param}" ${ncs_install_type}
            exit_install_log 0
        else
            log "ERROR" "Ncs package install failed, please retry after uninstall!"
            log_operation "${operation}" "failed !"
            exit_install_log 1
        fi
    fi
    return $?
}

# 升级调用子脚本
upgrade_run() {
    local operation="Upgrade"
    if [ -f "$install_info" ]; then
        ncs_input_install_path=$(get_install_param "ncs_install_path_param" "${install_info}")
        local ncs_install_type=$(get_install_param "ncs_install_type" "${install_info}")
    elif [ -f "$install_info_old" ] && [ $(grep -c -i "ncs_install_path_param" $install_info_old) -ne 0 ]; then
        . $install_info_old
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete ${install_info} or ${install_info_old}"
        exit_install_log 1
    fi
    if [[ "${is_docker_install}" == y ]] ; then
        local ncs_install_path_param=$(concat_docker_install_path "${docker_root}" "${ncs_input_install_path}")
    else
        local ncs_install_path_param=${ncs_input_install_path}
    fi
    if [ "$pkg_is_multi_version" = "true" ]; then
        ncs_install_path_param="${ncs_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
    else
        ncs_install_path_param="${ncs_install_path_param}/${pkg_relative_path}"
    fi
    if [ "$1" = "upgrade" ]; then
        chmod_start
        new_echo "INFO" "upgrade ${ncs_install_path_param} ${ncs_install_type}"
        log "INFO" "upgrade ${ncs_install_path_param} ${ncs_install_type}"
        bash ${curpath}/run_ncs_upgrade.sh "upgrade" "$ncs_input_install_path" "$ncs_install_type" "$is_quiet" \
            "$input_setenv" "${is_docker_install}" "$docker_root" "$in_install_for_all"
        if [ $? -eq 0 ]; then
            log "INFO" "Ncs package upgraded successfully! The new version takes effect immediately."
            log_operation "${operation}" "success"
            chmod_end
            set_env_var "${ncs_install_path_param}" ${ncs_install_type}
            exit_install_log 0
        else
            log "ERROR" "Ncs package upgrade failed, please retry after uninstall!"
            log_operation "${operation}" "failed !"
            exit_install_log 1
        fi
    fi
    return $?
}

# 卸载调用子脚本
uninstall_run() {
    local pkg_install_path="$2"
    local is_multi_version="$3"
    local uninstall="$4"
    local install_info="${pkg_install_path}/ascend_install.info"
    local operation="Uninstall"
    if [ ! -f "${pkg_install_path}/script/run_ncs_uninstall.sh" ]; then
        log "WARNING" "run_ncs_uninstall.sh not found."
        return $?
    fi
    local ncs_install_type
    if [ -f "$install_info" ]; then
        ncs_install_type=$(get_install_param "ncs_install_type" "${install_info}")
        local ncs_input_install_path=$(get_install_param "ncs_install_path_param" "${install_info}")
    elif [ -f "$install_info_old" ] && [ $( grep -c -i "ncs_install_path_param" $install_info_old ) -ne 0 ]; then
        . $install_info_old
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete ${install_info} or ${install_info_old}"
        exit_install_log 1
    fi

    if [ "${is_docker_install}" = y ] ; then
        local ncs_install_path_param=$(concat_docker_install_path "${docker_root_install_path}" "${ncs_input_install_path}")
    else
        local ncs_install_path_param=${ncs_input_install_path}
    fi
    if [ "$is_multi_version" = "true" ]; then
        ncs_install_path_param="${ncs_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
    else
        ncs_install_path_param="${ncs_install_path_param}/${pkg_relative_path}"
    fi

    if [ "$1" = "uninstall" ]; then
        chmod_start
        new_echo "INFO" "uninstall ${ncs_install_path_param} ${ncs_install_type}"
        log "INFO" "uninstall ${ncs_install_path_param} ${ncs_install_type}"
        bash ${ncs_install_path_param}/script/run_ncs_uninstall.sh "uninstall" "$ncs_input_install_path" $ncs_install_type $is_quiet $docker_root_install_path
        if [ $? -eq 0 ]; then
            if [ "$uninstall" = y ]; then
                test -f "$install_info" && rm -f "$install_info"
            fi
            if [ "$uninstall" = y ]; then
                remove_dir_recursive $ncs_input_install_path $ncs_install_path_param
            fi

            if [ "$uninstall" = y ] && [ "$(ls -A "$ncs_input_install_path")" = "" ]; then
                [ -n "${ncs_input_install_path}" ] && rm -rf "$ncs_input_install_path"
            fi

            if [ $(id -u) -eq 0 ]; then
                if [ "$uninstall" = y ] && [ -f "$install_info_old" ] && [ $( grep -c -i "ncs_install_path_param" $install_info_old ) -ne 0 ]; then
                    sed -i '/ncs_install_path_param=/d' $install_info_old
                    sed -i '/ncs_install_type=/d' $install_info_old
                fi
            fi
            new_echo "INFO" "Ncs package uninstalled successfully! Uninstallation takes effect immediately."
            log "INFO" "Ncs package uninstalled successfully! Uninstallation takes effect immediately."
            log_operation "${operation}" "success"
        else
            log "ERROR" "Ncs package uninstalled failed !"
            log_operation "${operation}" "failed !"
            exit_uninstall_log 1
        fi
    fi
    return $?
}

save_user_files_to_log(){
    if [ "$1" = "${default_dir}" ] && [ -s "$1" ]; then
        local file_num=$(ls -lR "$1"|grep "^-"|wc -l)
        local dir_num=$(ls -lR "$1"|grep "^d"|wc -l)
        local total_num=$(expr ${file_num} + ${dir_num})
        if [ $total_num -eq 2 ]; then
            if [ -f "${install_info}" ] && [ -f "${default_dir}"/version.info ]; then
                return 0
            fi
        fi
        if [ $total_num -eq 1 ]; then
            if [ -f "${install_info}" ] || [ -f "${default_dir}"/version.info ]; then
                return 0
            fi
        fi
        log "INFO" "Some files generated by user are not cleared, if necessary, manually clear them, get details in $log_file"
    fi
    if [ -s "$1" ]; then
        for file in $(ls -a "$1"); do
            if test -d "$1/$file"; then
                if [[ "$file" != '.' && "$file" != '..' ]]; then
                    echo "$1/$file" >> $log_file
                    save_user_files_to_log "$1/$file"
                fi
            else
                echo "$1/$file" >> $log_file
            fi
        done
    fi
}

judgement_path() {
    . "${common_func_path}"
    check_install_path_valid "${1}"
    if [ $? -ne 0 ];then
        log "ERROR" "The Ncs install_path ${1} is invalid, only [a-z,A-Z,0-9,-,_] is support!"
        exit 1
    fi
}

unique_mode() {
    if [ ! -z "$g_param_check_flag" ]; then
        log "ERROR" "ERR_NO:0x0004;ERR_DES:only support one type: full/run/docker/devel/upgrade/uninstall, operation failed!"
        exit_install_log 1
    fi
}

check_install_for_all() {
    local mod_num
    if [ "$input_install_for_all" = y ] && [ -d "$install_path" ]; then
        mod_num=$(stat -c %a ${install_path})
        mod_num=$(check_chmod_length $mod_num)
        other_mod_num=${mod_num:2:3}
        if [ "${other_mod_num}" -ne 5 ] && [ "${other_mod_num}" -ne 7 ]; then
            log "ERROR" "${install_path} permission is ${mod_num}, this permission does not support install_for_all param."
            exit_install_log 1
        fi
    fi
}

pre_check() {
    local check_shell_path=${curpath}/../bin/prereq_check.bash
    if [ ! -f "${check_shell_path}" ]; then
        log "WARNING" "${check_shell_path} not exist."
        return 0
    fi

    if [ ! x"$quiet" = "x" ] && [ "$quiet" = y ]; then
        bash "${check_shell_path}" --quiet
    else
        bash "${check_shell_path}" --no-quiet
    fi
}

migrate_user_assets_v2() {
    if [ "$pkg_is_multi_version" = "true" ]; then
        get_package_last_installed_version_dir "last_installed_dir" "$install_path" "${PACKAGE_NAME}"
        if [ -n "$last_installed_dir" ]; then
            last_installed_dir="$install_path/$last_installed_dir"
            local data_dir="${pkg_relative_path}/conf"
            merge_config "$last_installed_dir/$data_dir/Ncs.ini" "${curpath}/../conf/Ncs.ini"
        fi
    fi
}

# 创建子包安装目录
create_default_dir() {
    if [ ! -d "$default_dir" ]; then
        create_install_dir "$default_dir" "${username}":"${usergroup}"
    fi
    if [ -n "$pkg_version_dir" ]; then
        create_install_dir "$(dirname $default_dir)" "$username:$usergroup"
        create_install_dir "$(dirname $(dirname $default_dir))" "$username:$usergroup"
    fi
    [ -d "$default_dir" ] && return 0
    return 1
}

#runfile=$(expr substr $2 3 $(expr ${#2} - 2))/$(expr substr $1 5 $(expr ${#1} - 4))
runfilename=$(expr substr "$1" 5 $(expr ${#1} - 4))

full_install=n
run_install=n
docker_install=n
devel_install=n
uninstall=n
upgrade=n
install_mode=""
install_path_cmd="--install-path"
input_install_path=""
in_install_for_all=""
docker_root=""
setenv=""
input_path_flag=n
input_install_for_all=n
is_docker_install=n
input_pre_check=n
input_setenv=n
uninstall_path_cmd="--uninstall"
uninstall_path_param=""
upgrade_path_cmd="--upgrade"
upgrade_path_param=""
docker_cmd="--docker"
chip_flag=n
chipmode="all"
feature_flag=n
featuremode="all"
is_quiet=n
is_check=n
install_path_param="/usr/local/Ascend"
host_os_name=unknown
host_os_version=unknown
arch_name=unknown
g_param_check_flag=""

# 设置默认安装参数
is_input_path="n"
in_install_path=""

if [ $(id -u) -eq 0 ]; then
    input_install_for_all=y
    in_install_for_all="--install_for_all"
else
    install_path_param=$default_normal_dir
fi
####################################################################################################

if [[ "$#" == "1" ]] || [[ "$#" == "2" ]]; then
    log "ERROR" "ERR_NO:0x0004;ERR_DES:Unrecognized parameters. Try './xxx.run --help for more information.'"
    exit_install_log 1
fi

i=0
while true
do
    if [ x"$1" = x"" ];then
        break
    fi
    if [ "$(expr substr ""$1"" 1 2 )" = "--" ]; then
        i=$(expr $i + 1)
    fi
    if [ $i -gt 2 ]; then
        break
    fi
    shift 1
done

start_time=$(date +"%Y-%m-%d %H:%M:%S")
all_parma="$@"


while true
do
    case "$1" in
    --help | -h)
        param_usage
        exit_install_log 0
        ;;
    --run)
        unique_mode
        g_param_check_flag="True"
        run_install=y
        install_mode="run"
        shift
        ;;
    --full)
        unique_mode
        g_param_check_flag="True"
        full_install=y
        install_mode="full"
        shift
        ;;
    --docker)
        unique_mode
        g_param_check_flag="True"
        docker_install=y
        install_mode="docker"
        shift
        ;;
    --devel)
        unique_mode
        g_param_check_flag="True"
        devel_install=y
        install_mode="devel"
        shift
        ;;
    --install-path=*)
        temp_path=$(echo "$1" | cut -d"=" -f2-)
        judgement_path "${temp_path}"
        slashes_num=$( echo "${temp_path}" | grep -o '/' | wc -l)
        # 去除指定安装目录后所有的 "/"
        if [ $slashes_num -gt 1 ];then
            input_install_path=$(echo "${temp_path}" | sed "s/\/*$//g")
        else
            input_install_path="${temp_path}"
        fi
        input_path_flag=y
        shift
        ;;
    --chip=*)
        chip_flag=y
        shift
        ;;
    --feature=*)
        featuremode=$(echo "$1" | cut -d"=" -f2-)
        feature_flag=y
        shift
        ;;
    --install-for-all)
        input_install_for_all=y
        in_install_for_all="--install_for_all"
        shift
        ;;
    --docker-root=*)
        is_docker_install=y
        docker_root=$(echo "$1" | cut -d"=" -f2-)
        check_docker_path $docker_root
        shift
        ;;
    --pre-check)
        input_pre_check=y
        shift
        ;;
    --setenv)
        input_setenv=y
        setenv="--setenv"
        shift
        ;;
    --uninstall)
        unique_mode
        g_param_check_flag="True"
        uninstall=y
        shift
        ;;
    --upgrade)
        unique_mode
        g_param_check_flag="True"
        upgrade=y
        shift
        ;;
    --quiet)
        is_quiet=y
        shift
        ;;
    --extract=*)
        shift;
        ;;
    --keep)
        shift;
        ;;
    --check)
        shift
        exit 0
        ;;
    --version)
        get_version_run_file
        version=y
        exit 0
        ;;
    -*)
        log "ERROR" "ERR_NO:0x0004;ERR_DES: Unsupported parameters : $1"
        param_usage
        exit_install_log 0
        ;;
    *)
        break
        ;;
    esac
done

architecture=$(uname -m)
# check platform
if [ "${architecture}" != "${platform_data}" ] ; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:The architecture of the run package \
is inconsistent with that of the current environment. "
    exit 1
fi

## --quiet不支持单独使用校验
if [ "${is_quiet}" = "y" ];then
    if [ "${upgrade}" = "y" ] || [ "${full_install}" = "y" ] || [ "${run_install}" = "y" ] || [ "${devel_install}" = "y" ] || [ "${uninstall}" = "y" ];then
        is_quiet=y
    else
        log "WARNING" "'--quiet' is not supported to used by this way,please use with '--full','--devel','--run' or '--upgrade','--uninstall'"
        exit 1
    fi
fi

username="${default_normal_username}"
usergroup="${default_narmal_usergroup}"

######################  check params confilct ###################
# part1: 明确install_path
if [ "$full_install" = y ] || [ "$run_install" = y ] || [ "$devel_install" = y ] || [ "${upgrade}" = y ] || [ "${uninstall}" = y ]; then
    if [ "$input_path_flag" = y ]; then
        input_install_path=$(relative_path_to_absolute_path "${input_install_path}")
        input_install_path=$(get_install_path)
        install_path_param="${input_install_path}"
    fi
fi

operation_logfile="${log_dir}/operation.log"
log_file="${log_dir}/ascend_install.log"
if [[ "${is_docker_install}" == y ]]; then
    install_path=$(concat_docker_install_path "${docker_root}" "${install_path_param}")
    operation_logfile=$(concat_docker_install_path "${docker_root}" "${operation_logfile}")
    log_file=$(concat_docker_install_path "${docker_root}" "${log_file}")
    default_root_dir=$(concat_docker_install_path "${docker_root}" "${default_root_dir}")
else
    install_path=${install_path_param}
fi

# part3: 明確多版本
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
get_version_dir "pkg_version_dir" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ]; then
    default_dir="${install_path}/$pkg_version_dir/${pkg_relative_path}"
else
    default_dir="${install_path}/${pkg_relative_path}"
fi

log_dirs=$(concat_docker_install_path "${docker_root}" "${log_dir}")
create_folder $log_dirs
change_log_mode $log_file
install_info="${default_dir}/ascend_install.info"
if [ "$uninstall" = y ]; then
    start_uninstall_log
else
    start_install_log
fi

# 检查chip参数是否冲突
if [ "${chip_flag}" = "y" ] && [ "${uninstall}" = "y" ]; then
    log "ERROR" "'--chip' is not supported to used by this way, please use with '--full', '--devel', '--run', '--upgrade'"
    exit 1
fi

# 检查feature参数是否冲突
if [ "${feature_flag}" = "y" ] && [ "${uninstall}" = "y" ]; then
    log "ERROR" "'--feature' is not supported to used by this way, please use with '--full', '--devel', '--run', '--upgrade'"
    exit 1
fi

if [ "$featuremode" != "all" ]; then
    contain_feature "ret" "$featuremode" "$curpath/filelist.csv"
    if [ "$ret" = "false" ]; then
        log "WARNING" "Ncs package doesn't contain features $featuremode, skip installation."
        exit 0
    fi
fi

# pre-check
if [ "${input_pre_check}" = y ]; then
    log "INFO" "Ncs do pre check started."
    pre_check
    if [ $? -ne 0 ]; then
        log "WARNING" "Ncs do pre check failed."
    else
        log "INFO" "Ncs do pre check finished."
    fi
    if [ "$full_install" = n ] && [ "$run_install" = n ] && [ "$devel_install" = n ] && [ "${upgrade}" = n ] && [ "${uninstall}" = n ]; then
        exit_install_log 0
    fi
fi

######################  check params confilct ###################
# 卸载参数只支持单独使用
if [ "$uninstall" = y ]; then
    if [ "${upgrade}" = y ] || [ "$full_install" = y ] || [ "$run_install" = y ] || [ "$docker_install" = y ] || [ "$devel_install" = y ]; then
        log "ERROR" "ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
        exit_uninstall_log 1
    fi
fi

##################################################################
# 安装升级运行态时，1/2包必须已安装，且指定的用户必须存在且与1/2包同属组
if [ "$input_install_for_all" = n ]; then
    if [ "$run_install" = y ] || [ "$full_install" = y ]; then
        confirm=n
        base_installinfo="/etc/ascend_install.info"
        if [ ! -f "$base_installinfo" ]; then
            log "WARNING" "driver and firmware is not exists,please install first."
            confirm=y
        elif [ $(grep -c -i "Driver" ${base_installinfo}) -eq 0 ]; then
            log "WARNING" "driver is not exists,please install first."
            confirm=y
        elif [ $(grep -c -i "Firmware" ${base_installinfo}) -eq 0 ]; then
            log "WARNING" "firmware is not exists,please install first.(docker scenes is not need)"
            confirm=y
        else
            usergroup_base=$(grep -i UserGroup= "${base_installinfo}" | cut -d"=" -f2-)
            check_group "${usergroup_base}" "${username}"
            if [ "$?" != 0 ]; then
                log "ERROR" "ERR_NO:0x0093;ERR_DES:User is not belong to the dirver or firmware's installed usergroup!Please add the user (${username}) to the group (${usergroup_base})."
                confirm=y
                exit_install_log 1
            fi
        fi
    fi
fi

check_install_for_all
create_default_install_dir_for_commonuer
log_base_version
username=$(get_user_name)
usergroup=$(get_user_group)
is_valid_path

if [ "$full_install" = y ] || [ "$run_install" = y ] || [ "$devel_install" = y ]; then
    if [ ! -d "$default_dir" ]; then
        create_default_dir
    fi
fi

# 环境上是否已安装过run包
if [ -f "${curpath}"/install_common_parser.sh ]; then
    flag_install=$(bash "${curpath}"/install_common_parser.sh --pkg-in-dbinfo --package=ncs "${install_path}")
fi

version2=$(get_version_installed "${default_dir}")
if [ "$version2""x" != "x" -a "$version2" != "none" ] || [ -f "${install_info}" ]; then
    # 卸载场景
    if [ "$uninstall" = y ]; then
        uninstall_run "uninstall" ${default_dir} ${pkg_is_multi_version} $uninstall
        save_user_files_to_log "$default_dir"
    # 升级场景
    elif [ "$upgrade" = y ]; then
        if [ -n "$pkg_version_dir" ]; then
            get_package_upgrade_version_dir "upgrade_version_dir" "$install_path" "ncs"
            if [ -z "$upgrade_version_dir" ]; then
                log "ERROR" "Can not find softlink for this package in latest directory, upgrade failed"
                log_operation "Upgrade" "failed"
                exit_install_log 1
            elif [ "$upgrade_version_dir" != "$pkg_version_dir" ]; then
                uninstall_script="$install_path/$upgrade_version_dir/$pkg_relative_path/script/uninstall.sh"
                if [ -f "$uninstall_script" ]; then
                    $uninstall_script
                fi
            fi
        fi
        uninstall_run "uninstall" ${default_dir} ${pkg_is_multi_version} $uninstall
        save_user_files_to_log "$default_dir"
        upgrade_run "upgrade"
    # 安装场景
    elif [ "$run_install" = y ] || [ "$full_install" = y ] || [ "$devel_install" = y ]; then
        # run模式和full模式 安装场景
        version1=$(get_version_run_file)
        # 判断是否要覆盖式安装
        if [ "$is_quiet" = n ]; then
            if [ "${input_install_path}" = "" ]; then
                path_=${default_root_dir}
            elif [ "${input_install_path}" != "" ]; then
                path_=${install_path}
            fi
            log "INFO" "Ncs package has been installed on the path ${path_}, the version is ${version2}, and the version of this package is ${version1}, do you want to continue?  [y/n] "
            while true
            do
                read yn
                if test "$yn" = n; then
                    echo "stop installation!"
                    chattr_files ${path_}
                    exit_install_log 0
                elif test "$yn" = y; then
                    break
                else
                    log "ERROR" "ERR_NO:0x0002;ERR_DES:input error, please input again!"
                fi
            done
        fi
        uninstall_run "uninstall" ${default_dir} ${pkg_is_multi_version} $uninstall
        save_user_files_to_log "$default_dir"
        if [ ! -d "$default_dir" ]; then
           mkdir "$default_dir"
        fi

        install_run "install"
    fi
else
    # 卸载场景
    if [ "$uninstall" = y ]; then
        if [[ -d "$default_dir" ]]; then
            log "ERROR" "The current user does not have the required permission to uninstall $default_dir, uninstall failed"
            log_operation "Uninstall" "failed"
            exit_uninstall_log 1
        else
            log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed on ${install_path}, uninstall failed"
            log_operation "Uninstall" "failed"
            if [ "$(ls -A "$install_path_param")" = "" ]; then
                test -d "$install_path_param" && rm -rf "$install_path_param"
            fi
            exit_uninstall_log 1
        fi
    # 升级场景
    elif [ "$upgrade" = y ]; then
        if [ -z "$pkg_version_dir" ]; then
            if [ -d "$default_dir" ]; then
                log "ERROR" "The current user does not have the required permission to uninstall $default_dir, upgrade failed"
                log_operation "Upgrade" "failed"
                exit_install_log 1
            else
                log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed in ${install_path}, upgrade failed"
                log_operation "Upgrade" "failed"
                if [ "$(ls -A "$install_path_param")" = "" ]; then
                    test -d "$install_path_param" && rm -rf "$install_path_param"
                fi
                exit_install_log 1
            fi
        else
            get_package_upgrade_version_dir "upgrade_version_dir" "$install_path" "ncs"
            if [ -n "$upgrade_version_dir" ]; then
                last_install_info="$install_path/$upgrade_version_dir/$pkg_relative_path/ascend_install.info"
                create_default_dir && cp "$last_install_info" "$default_dir"
                migrate_user_assets_v2
                uninstall_script="$(dirname $last_install_info)/script/uninstall.sh"
                if [ -f "$uninstall_script" ]; then
                    $uninstall_script
                fi
                upgrade_run "upgrade"
                exit_install_log 0
            else
                log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed in ${install_path}, upgrade failed"
                log_operation "Upgrade" "failed"
                if [ "$(ls -A "$install_path")" = "" ]; then
                    test -d "$install_path" && rm -rf "$install_path"
                fi
                exit_install_log 1
            fi
        fi
    # 安装场景
    elif [ "$run_install" = y ] || [ "$full_install" = y ] || [ "$devel_install" = y ]; then
        install_run "install"
        exit_install_log 0
    fi
fi

if [ "$uninstall" = y ]; then
    exit_uninstall_log 0
fi
