#!/bin/bash
# The main script of installing aoe.
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

default_root_dir="/usr/local/Ascend"
default_normal_dir="${HOME}/Ascend"
default_normal_username=$(id -nu)
default_normal_usergroup=$(id -ng)
username="${default_normal_username}"
usergroup="${default_normal_usergroup}"
curpath=$(dirname $(readlink -f $0))
common_func_path="${curpath}/common_func.inc"
common_func_v2_path="${curpath}/common_func_v2.inc"
version_cfg_path="${curpath}/version_cfg.inc"
pkg_version_path="${curpath}/../../version.info"
pkg_info_file="${curpath}/../scene.info"
platform_data=$(grep -e "arch" "$pkg_info_file" | cut --only-delimited -d"=" -f2-)
COMMON_SHELL="${curpath}/common.sh"
install_info_old="/etc/ascend_install.info"
RUN_DIR="$(echo $2 | cut -d'-' -f 3)"
TARGET_INSTALL_PATH=""
TARGET_USERNAME=""
TARGET_USERGROUP=""

. "${common_func_path}"
. "${common_func_v2_path}"
. "${version_cfg_path}"

# load shell
source "${COMMON_SHELL}"

VERSION_INFO="version.info"

if [ $(id -u) -ne 0 ]; then
    default_root_dir=${default_normal_dir}
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

# 递归授权
chmod_recur() {
    local file_path="${1}"
    local permision="${2}"
    local type="${3}"
    local permission=$(setfile_chmod $permision)
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
    local permission=$(setfile_chmod $permision)
    if [ "$type" = "dir" ]; then
        chmod $permission "$file_path"
    elif [ "$type" = "file" ]; then
        chmod $permission "$file_path"
    fi
}

#设置权限
setfile_chmod() {
    local _permission="${1}"
    local _new_permission
    if [[ "${input_install_for_all}" = "y" ]]; then
	    _new_permission="$(expr substr $_permission 1 2)$(expr substr $_permission 2 1)"
        echo $_new_permission
    else
        echo $_permission
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
    chmod_single_dir "${default_dir}" 550 dir 2> /dev/null
    chmod_recur "${default_dir}/script" 500 file 2> /dev/null
    chmod_single_dir "${default_dir}/script" 500 dir 2> /dev/null
    chown -Rf "$username":"$usergroup" "${default_dir}" 2> /dev/null
    if [ $(id -u) -eq 0 ]; then
        chown "root:root" "$default_dir" 2> /dev/null
        chmod 755 "$default_dir" 2> /dev/null
        chown -R "root:root" "${default_dir}/script" 2> /dev/null
    fi
}

param_usage() {
    log "INFO" "Please input this command for help: ./${runfilename} --help"
}

# 修改日志文件的权限
change_log_mode() {
    if [ ! -f "$log_file" ]; then
        touch $log_file
    fi
    chmod_single_dir $log_file 640 file
}

# 创建文件夹
create_log_folder() {
    if [ ! -d "$log_dir" ]; then
        mkdir -p $log_dir
    fi
    if [ $(id -u) -ne 0 ]; then
        chmod_single_dir $log_dir 740 dir
    else
        chmod_single_dir $log_dir 750 dir
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
    log "INFO" "OperationLogFile:${operation_log_file}"
}
# 开始卸载前打印开始信息
start_uninstall_log() {
    local cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    new_echo "INFO" "Start time:${cur_date}"
    log "INFO" "Start time:$cur_date"
    log "INFO" "LogFile:${log_file}"
    log "INFO" "InputParams:$all_parma"
    log "INFO" "OperationLogFile:${operation_log_file}"
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

    if [ ! -f "${operation_log_file}" ]; then
        touch ${operation_log_file}
        chmod_single_dir ${operation_log_file} 640 file
    fi

    echo "$1 $level root $cur_date 127.0.0.1 $runfilename $2 installmode=$installmode; cmdlist=$all_parma" >> $operation_log_file
}

#相对路径转化绝对路径
relative_path_to_absolute_path() {
    local relative_path_="${1}"
    local fstr="$(expr substr "$relative_path_" 1 1)"
    if [ "$fstr" = "./" ];then
        local path="${RUN_DIR}/${relative_path_}"
        echo "${path}"
    else
        echo "${1}"
    fi
}

change_install_path() {
    local _home=""
    local _prefix=""
    local _install_dir=""
    local _tmp_path="${1}"
    _tmp_path=$(echo "${_tmp_path}" | sed "s/\/*$//g")
    if [ x"${_tmp_path}" = "x" ]; then
        _tmp_path="/"
    fi

    _prefix=$(echo "${_tmp_path}" | cut -d"/" -f1 | cut -d"~" -f1)
    if [ x"${_prefix}" = "x" ]; then
        _install_dir="${_tmp_path}"
    else
        _prefix=$(echo "${RUN_DIR}" | cut -d"/" -f1 | cut -d"~" -f1)
        if [ x"${_prefix}" = "x" ]; then
            _install_dir="${RUN_DIR}/${_tmp_path}"
        else
            log "ERROR" "ERR_NO:0x0004;ERR_DES: Run package path is invalid: $RUN_DIR"
            exitLog 1
        fi
    fi

    _home=$(echo "${_install_dir}" | cut -d"~" -f1)
    if [ "x${_home}" = "x" ]; then
        _tmp_path=$(echo "${_install_dir}" | cut -d"~" -f2)
        if [ "$(id -u)" -eq 0 ]; then
            _install_dir="/root${_tmp_path}"
        else
            _install_dir="/home/$(whoami)${_tmp_path}"
        fi
    fi
    echo "${_install_dir}"
}

# get install path
get_install_path() {
    if [ "$input_path_flag" = y ]; then
        if [ x"${input_install_path}" = "x" ]; then
            log "ERROR" "ERR_NO:0x0004;ERR_DES: Install path is empty."
            exitLog 1
        fi

        install_dir=$(relative_path_to_absolute_path "${input_install_path}")
        install_dir=$(change_install_path "${install_dir}")
        if [ x"${install_dir}" = "x" ]; then
            exitLog 1
        fi
    else
        # use default path
        if [ $(id -u) -eq 0 ]; then
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
            exitLog 1
        fi
    fi
}

create_install_dir() {
    local path=$1
    local user_and_group=$2
    local permission

    if [ $(id -u) -eq 0 ]; then
        user_and_group="root:root"
        permission=755
    else
        permission=750
    fi

    if [ "$input_install_for_all" = y ]; then
        permission=755
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
        [ -n "${_file}" ] && rm -rf "${_file}"
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
    input_install_path_param=""
    if [ ! "${input_install_path}" = "" ]; then
        if [ ! -d "${input_install_path}" ]; then
            local up_dir=$(dirname "${input_install_path}")
            if [ "${input_install_path}" = "${default_root_dir}" ] || [ "${input_install_path}" = "${default_normal_dir}" ] || [ -d "${up_dir}" ];then
                create_install_dir "${input_install_path}" "${username}":"${usergroup}"
                input_install_path_param=$(cd "${input_install_path}";pwd)
            else
                # 路径不存在，报错
                log "ERROR" "ERR_NO:0x0003;ERR_DES:The $input_install_path dose not exist, please retry a right path."
                exit_install_log 1
            fi
        else
            local ret=0
            if [ $(id -u) -eq 0 ]; then
                parent_dirs_permision_check "${input_install_path}" && ret=$? || ret=$?
                if [ "${is_quiet}" = y ] && [ ${ret} -ne 0 ]; then
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
            input_install_path_param=$(cd "${input_install_path}";pwd)
            if [ $(id -u) -ne 0 ]; then
                #cd $input_install_path_param >> /dev/null 2>&1
                log "DEBUG" "$input_install_path_param"
            else
                sh -c 'cd "$input_install_path_param" >> /dev/null 2>&1'
            fi
            if [ ! $? = 0 ]; then
                log "ERROR" "ERR_NO:0x0093;ERR_DES:The $username do not have the permission to access $input_install_path_param, please reset the directory to a right permission."
                exit_install_log 1
            fi
        fi
    fi
}

parent_dirs_permision_check() {
    current_dir="$1" parent_dir="" short_install_dir=""
    local owner="" mod_num=""

    parent_dir=$(dirname "${current_dir}")
    short_install_dir=$(basename "${current_dir}")

    if [ "${current_dir}"x = "/"x ]; then
        log "INFO" "parent_dirs_permision_check success"
        return 0
    else
        owner=$(stat -c %U "${parent_dir}"/"${short_install_dir}")
        if [ "${owner}" != "root" ]; then
            log "WARNING" "[${short_install_dir}] permision isn't right, it should belong to root."
            return 1
        fi

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

#安装路径加锁
chattr_files() {
    local _install_path=$1
    if [ -f "$install_info" ]; then
        #安装路径存在时加锁
        if [ -d "${_install_path}/${pkg_relative_path}" ]; then
            chattr -R +i "${_install_path}/${pkg_relative_path}" >> /dev/null 2>&1
            if [ $? -ne 0 ];then
                log "DEBUG" "chattr +i for the subfiles."
                find ${_install_path}/${pkg_relative_path} -name "*" | xargs chattr +i  >> /dev/null 2>&1
            else
                log "DEBUG" "chattr -R +i $_install_path succeeded."
            fi
        fi
    fi
}

#解锁
unchattr_files() {
    local _install_path=$1
    if [ -f "$install_info" ]; then
        if [ -d "${_install_path}" ]; then
            chattr -R -i "${_install_path}" >> /dev/null 2>&1
            if [ $? -ne 0 ];then
                log "DEBUG" "unchattr -i for the subfiles."
                find ${_install_path} -name "*" | xargs chattr -i  >> /dev/null 2>&1
            else
                log "DEBUG" "unchattr -R -i $_install_path succeeded."
            fi
        fi
    fi
}

#创建普通用户的默认安装目录
create_default_install_dir_for_common_user() {
    if [ $(id -u) -ne 0 ]; then
        if [ "$input_path_flag" = n ]; then
            if [ ! -d "$install_path_param" ]; then
                create_install_dir "${install_path}" "${username}:${usergroup}"
            fi
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

# 获取安装目录下的完整版本号  version2
get_version_installed() {
    local aoe_version="none"
    if [ -f "$1/${VERSION_INFO}" ]; then
        aoe_version="$(grep -iw version "$1/${VERSION_INFO}" | cut -d"=" -f2-)"
    fi
    echo "$aoe_version"
}

# 获取run包中的完整版本号  version1
get_version_in_run_file() {
    local aoe_version="none"
    if [ -f "${pkg_version_path}" ]; then
        aoe_version="$(grep -iw version "${pkg_version_path}" | cut -d"=" -f2-)"
    fi
    echo "$aoe_version"
}

log_base_version() {
    if [ -f "${install_info}" ];then
        installed_version=$(get_version_installed "${default_dir}")
        if [ ! "${installed_version}"x = ""x ]; then
            log "INFO" "base version is ${installed_version}."
            return 0
        fi
    fi
    log "WARNING" "base version was destroyed or not exist."
}

set_install_path_param() {
    install_path_param="${1}"
}

update_install_path() {
    if [ ! "$input_install_path" = "" ]; then
        set_install_path_param "${input_install_path}"
    fi

    if [ ! -d "${install_path}" ]; then
        # 路径不存在，报错
        log "ERROR" "ERR_NO:0x0003;ERR_DES:The $install_path dose not exist, please retry a right path."
        exit_install_log 1
    fi
}

update_install_info() {
    if [ -f "$install_info" ]; then
        local aoe_install_type=$(get_install_param "Aoe_Install_Type" "${install_info}")
        local aoe_user_name=$(get_install_param "Aoe_UserName" "${install_info}")
        local aoe_user_group=$(get_install_param "Aoe_UserGroup" "${install_info}")
        local aoe_install_path_param=$(get_install_param "Aoe_Install_Path_Param" "${install_info}")
        if [ -z "$aoe_install_type" ]; then
            update_install_param "Aoe_Install_Type" "${installmode}" "${install_info}"
        else
            chmod_single_dir "${aoe_install_path_param}" 755 dir
            update_install_param "Aoe_Install_Type" "${installmode}" "${install_info}"
        fi

        if [ -z "${aoe_user_name}" ]; then
            update_install_param "Aoe_UserName" "${username}" "${install_info}"
        fi

        if [ -z "${aoe_user_group}" ]; then
            update_install_param "Aoe_UserGroup" "${usergroup}" "${install_info}"
        fi

        if [ -z "${aoe_install_path_param}" ]; then
            update_install_param "Aoe_Install_Path_Param" "${install_path_param}" "${install_info}"
        else
            chmod_single_dir "${aoe_install_path_param}/" 755 dir
            update_install_param "Aoe_Install_Path_Param" "${install_path_param}" "${install_info}"
        fi
    else
        create_file "$install_info" "${username}":"${usergroup}" 600
        update_install_param "Aoe_Install_Type" "${installmode}" "${install_info}"
        update_install_param "Aoe_UserName" "${username}" "${install_info}"
        update_install_param "Aoe_UserGroup" "${usergroup}" "${install_info}"
        update_install_param "Aoe_Install_Path_Param" "${install_path_param}" "${install_info}"
    fi
}

set_environment_variable() {
    local _install_path="${1}"
    local _install_type=${2}
    if [ -n "$pkg_version_dir" ]; then
        _install_path="${_install_path}/${pkg_version_dir}"
    fi
    if [ "${_install_type}" = "full" ] || [ "${_install_type}" = "run" ] || [ "${_install_type}" = "upgrade" ]; then
    echo "Please make sure that
          - LD_LIBRARY_PATH includes ${_install_path}/${pkg_relative_path}/lib64"
    fi
}

concat_docker_install_path() {
    local _docker_path="$1"
    local _input_install_path="$2"
    local _install_path=""
    # delete last "/"
    _docker_path=$(echo "${_docker_path}" | sed "s/\/*$//g")
    if [ x"${_docker_path}" = "x" ]; then
        _docker_path="/"
    fi

    if [ "${_docker_path: -1}" = "/" ] && [ "${_input_install_path:0:1}" = "/" ]; then
         _docker_path=${_docker_path%?}
         _install_path=${_docker_path}${_input_install_path}
    elif [ "${_docker_path: -1}" != "/" ] && [ "${_input_install_path:0:1}" != "/" ]; then
        _install_path=${_docker_path}"/"${_input_install_path}
    else
        _install_path=${_docker_path}${_input_install_path}
    fi
    echo "${_install_path}"
}

# 安装调用子脚本
install_run() {
    local operation="Install"
    update_install_path
    update_install_info
    local aoe_input_install_path=$(get_install_param "Aoe_Install_Path_Param" "${install_info}")
    local aoe_install_type=$(get_install_param "Aoe_Install_Type" "${install_info}")
    if [ "${docker_root_install_flag}" = y ]; then
        local aoe_install_path_param=$(concat_docker_install_path "${docker_root_install_path}" "${aoe_input_install_path}")
    else
        local aoe_install_path_param=${aoe_input_install_path}
    fi
    if [ "$pkg_is_multi_version" = "true" ]; then
        aoe_install_path_param="${aoe_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
    else
        aoe_install_path_param="${aoe_install_path_param}/${pkg_relative_path}"
    fi

    unchattr_files ${aoe_install_path_param}
    if [ "$1" = "install" ]; then
        chmod_start
        new_echo "INFO" "install ${aoe_install_path_param} ${aoe_install_type}"
        log "INFO" "install ${aoe_install_path_param} ${aoe_install_type}"
        bash ${curpath}/run_aoe_install.sh "install" "$aoe_input_install_path" $aoe_install_type $is_quiet $input_install_for_all ${input_setenv} $docker_root_install_path
        if [ $? -eq 0 ]; then
            log "INFO" "Aoe package installed successfully! The new version takes effect immediately."
            log_operation "${operation}" "success"
            chmod_end
            set_environment_variable "${aoe_input_install_path}" ${aoe_install_type}
            exit_install_log 0
        else
            chmod_end
            log "ERROR" "Aoe package installed failed, please retry after uninstall!"
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
        local aoe_input_install_path=$(get_install_param "Aoe_Install_Path_Param" "${install_info}")
        local aoe_install_type=$(get_install_param "Aoe_Install_Type" "${install_info}")
    elif [ -f "$install_info_old" ] && [ $( grep -c -i "Aoe_Install_Path_Param" $install_info_old ) -ne 0 ]; then
        . $install_info_old
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete ${install_info} or ${install_info_old}"
        exit_install_log 1
    fi

    if [ "${docker_root_install_flag}" = y ] ; then
        local aoe_install_path_param=$(concat_docker_install_path "${docker_root_install_path}" "${aoe_input_install_path}")
    else
        local aoe_install_path_param=${aoe_input_install_path}
    fi
    if [ "$pkg_is_multi_version" = "true" ]; then
        aoe_install_path_param="${aoe_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
    else
        aoe_install_path_param="${aoe_install_path_param}/${pkg_relative_path}"
    fi

    if [ "$1" = "upgrade" ]; then
        chmod_start
        new_echo "INFO" "upgrade ${aoe_install_path_param} ${aoe_install_type}"
        log "INFO" "upgrade ${aoe_install_path_param} ${aoe_install_type}"
        bash ${curpath}/run_aoe_upgrade.sh "upgrade" "$aoe_input_install_path" $aoe_install_type $is_quiet $input_install_for_all ${input_setenv} $docker_root_install_path
        if [ $? -eq 0 ]; then
            log "INFO" "Aoe package upgraded successfully! The new version takes effect immediately."
            log_operation "${operation}" "success"
            chmod_end
            set_environment_variable "${aoe_input_install_path}" ${aoe_install_type}
            exit_install_log 0
        else
            chmod_end
            log "ERROR" "Aoe package upgraded failed, please retry after uninstall!"
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
    if [ ! -f "${pkg_install_path}/script/run_aoe_uninstall.sh" ]; then
        log "WARNING" "run_aoe_uninstall.sh not found."
        return $?
    fi
    local aoe_install_type
    if [ -f "$install_info" ]; then
        aoe_install_type=$(get_install_param "Aoe_Install_Type" "${install_info}")
        local aoe_input_install_path=$(get_install_param "Aoe_Install_Path_Param" "${install_info}")
    elif [ -f "$install_info_old" ] && [ $( grep -c -i "Aoe_Install_Path_Param" $install_info_old ) -ne 0 ]; then
        . $install_info_old
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete ${install_info} or ${install_info_old}"
        exit_install_log 1
    fi

    if [ "${docker_root_install_flag}" = y ] ; then
        local aoe_install_path_param=$(concat_docker_install_path "${docker_root_install_path}" "${aoe_input_install_path}")
    else
        local aoe_install_path_param=${aoe_input_install_path}
    fi
    if [ "$is_multi_version" = "true" ]; then
        aoe_install_path_param="${aoe_install_path_param}/${pkg_version_dir}/${pkg_relative_path}"
    else
        aoe_install_path_param="${aoe_install_path_param}/${pkg_relative_path}"
    fi

    if [ "$1" = "uninstall" ]; then
        chmod_start
        new_echo "INFO" "uninstall ${aoe_install_path_param} ${aoe_install_type}"
        log "INFO" "uninstall ${aoe_install_path_param} ${aoe_install_type}"
        bash ${aoe_install_path_param}/script/run_aoe_uninstall.sh "uninstall" "$aoe_input_install_path" $aoe_install_type $is_quiet $docker_root_install_path
        if [ $? -eq 0 ]; then
            if [ "$uninstall" = y ]; then
                test -f "$install_info" && rm -f "$install_info"
            fi
            if [ "$uninstall" = y ]; then
                remove_dir_recursive $aoe_input_install_path $aoe_install_path_param
            fi

            if [ "$uninstall" = y ] && [ "$(ls -A "$aoe_input_install_path")" = "" ]; then
                [ -n "${aoe_input_install_path}" ] && rm -rf "$aoe_input_install_path"
            fi

            if [ $(id -u) -eq 0 ]; then
                if [ "$uninstall" = y ] && [ -f "$install_info_old" ] && [ $( grep -c -i "Aoe_Install_Path_Param" $install_info_old ) -ne 0 ]; then
                    sed -i '/Aoe_Install_Path_Param=/d' $install_info_old
                    sed -i '/Aoe_Install_Type=/d' $install_info_old
                fi
            fi
            new_echo "INFO" "Aoe package uninstalled successfully! Uninstallation takes effect immediately."
            log "INFO" "Aoe package uninstalled successfully! Uninstallation takes effect immediately."
            log_operation "${operation}" "success"
        else
            log "ERROR" "Aoe package uninstalled failed !"
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
            if [ -f "${install_info}" ] && [ -f "${default_dir}/${VERSION_INFO}" ]; then
                return 0
            fi
        fi
        if [ $total_num -eq 1 ]; then
            if [ -f "${install_info}" ] || [ -f "${default_dir}/${VERSION_INFO}" ]; then
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

judgmentpath() {
    . "${common_func_path}"
    check_install_path_valid "${1}"
    if [ $? -ne 0 ];then
        log "ERROR" "The Aoe install_path ${1} is invalid, only [a-z,A-Z,0-9,-,_] is support!"
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
    if [ "$input_install_for_all" = y ] && [ -d "$install_path" ]; then
        local _mod_num=$(stat -c %a ${install_path})
        _mod_num=$(check_chmod_length ${_mod_num})
        local _other_mod_num=${_mod_num:2:3}
        if [ "${_other_mod_num}" -ne 5 ] && [ "${_other_mod_num}" -ne 7 ]; then
            log "ERROR" "${install_path} permission is ${_mod_num}, this permission does not support install_for_all param."
            exit_install_log 1
        fi
    fi
}

pre_check() {
    return 0
}

migrate_user_assets_v2() {
    if [ "$pkg_is_multi_version" = "true" ]; then
        get_package_last_installed_version_dir "last_installed_dir" "$install_path" "${PACKAGE_NAME}"
        if [ -n "$last_installed_dir" ]; then
            last_installed_dir="$install_path/$last_installed_dir"
            local data_dir="${pkg_relative_path}/conf"
            merge_config "$last_installed_dir/$data_dir/aoe.ini" "${curpath}/../conf/aoe.ini"
        fi
    fi
}

runfilename=$(expr substr "$1" 5 $(expr ${#1} - 4))

full_install=n
run_install=n
docker_install=n
docker_root_install_flag=n
devel_install=n
uninstall=n
upgrade=n
installmode=""

docker_root_install_path=""
input_install_path=""
input_path_flag=n
input_install_for_all=n
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
input_pre_check=n
input_setenv=n
set_install_path_param "${install_dir}"
host_os_name=unknown
host_os_version=unknown
arch_name=unknown
g_param_check_flag=""

# 二级命令
usr_input_secondary_cmd=""
install_path_cmd="--install-path"
docker_root_cmd="--docker-root"
install_for_all_cmd="--install-for-all"
setenv_cmd="--setenv"
is_quiet_cmd="--quiet"

# 设置默认安装参数
is_input_path="n"
in_install_path=""

if [[ "$(id -u)" == "0" ]]; then
    input_install_for_all=y
fi

####################################################################################################

if [[ "$#" == "1" ]] || [[ "$#" == "2" ]]; then
    log "ERROR" "ERR_NO:0x0004;ERR_DES:Unrecognized parameters. Try './xxx.run --help' for more information."
    exit_install_log 1
fi

i=0
while true
do
    if [ x"$1" = x"" ];then
        break
    fi
    if [ "$(expr substr "$1" 1 2 )" = "--" ]; then
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
        installmode="run"
        shift
        ;;
    --full)
        unique_mode
        g_param_check_flag="True"
        full_install=y
        installmode="full"
        shift
        ;;
    --docker)
        unique_mode
        g_param_check_flag="True"
        docker_install=y
        installmode="docker"
        shift
        ;;
    --docker-root=*)
        temp_path=$(echo "$1" | cut -d"=" -f2- )
        judgmentpath "${temp_path}"
        slashes_num=$( echo "${temp_path}" | grep -o '/' | wc -l )
        # 去除指定安装目录后所有的 "/"
        if [ $slashes_num -gt 1 ];then
            docker_root_install_path=$(echo "${temp_path}" | sed "s/\/*$//g")
        else
            docker_root_install_path="${temp_path}"
        fi
        docker_root_install_flag=y
        usr_input_secondary_cmd=$usr_input_secondary_cmd"'${docker_root_cmd}' "
        shift
        ;;
    --devel)
        unique_mode
        g_param_check_flag="True"
        devel_install=y
        installmode="devel"
        shift
        ;;
    --install-path=*)
        temp_path=$(echo "$1" | cut -d"=" -f2- )
        judgmentpath "${temp_path}"
        slashes_num=$( echo "${temp_path}" | grep -o '/' | wc -l )
        # 去除指定安装目录后所有的 "/"
        if [ $slashes_num -gt 1 ];then
            input_install_path=$(echo "${temp_path}" | sed "s/\/*$//g")
        else
            input_install_path="${temp_path}"
        fi
        input_path_flag=y
        usr_input_secondary_cmd=$usr_input_secondary_cmd"'${install_path_cmd}' "
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
        usr_input_secondary_cmd=$usr_input_secondary_cmd"'${install_for_all_cmd}' "
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
        usr_input_secondary_cmd=$usr_input_secondary_cmd"'${is_quiet_cmd}' "
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
    --pre-check)
        input_pre_check=y
        shift;
        ;;
    --setenv)
        input_setenv=y
        usr_input_secondary_cmd=$usr_input_secondary_cmd"'${setenv_cmd}' "
        shift;
        ;;
    --version)
        get_version_in_run_file
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

## --setenv --install-for-all --install-path=<path> --docker-root=<path> 不支持单独使用
if [ "${upgrade}" = "n" ] && [ "${full_install}" = "n" ] && [ "${run_install}" = "n" ] && [ "${devel_install}" = "n" ] && [ "${uninstall}" = "n" ]; then
    if [ -n "$usr_input_secondary_cmd" ];then
        echo "ERR_NO:0x0004;ERR_DES: ${usr_input_secondary_cmd}is not supported to be used by this way, Try './xxx.run --help' for more information."
        exit 1
    fi
fi
## --quiet不支持单独使用校验
if [ "${is_quiet}" = "y" ];then
    if [ "${upgrade}" = "y" ] || [ "${full_install}" = "y" ] || [ "${run_install}" = "y" ] || [ "${devel_install}" = "y" ] || [ "${uninstall}" = "y" ]; then
        is_quiet=y
    else
        log "ERROR" "ERR_NO:0x0004;ERR_DES: '--quiet' is not supported to be used by this way, Try './xxx.run --help' for more information."
        exit 1
    fi
fi

if [ "$uninstall" = y ];then
    username="${default_normal_username}"
    usergroup="${default_narmal_usergroup}"
fi

## log path
operation_log_file="${log_dir}/operation.log"
log_file="${log_dir}/ascend_install.log"

######################  check params confilct ###################
if [ "$full_install" = y ] || [ "$run_install" = y ] || [ "$devel_install" = y ] || [ "${upgrade}" = y ] || [ "${uninstall}" = y ]; then
    get_install_path
    input_install_path="${install_dir}"
    set_install_path_param "${input_install_path}"
fi

if [ "$docker_root_install_flag" = y ]; then
    docker_root_install_path=$(relative_path_to_absolute_path "${docker_root_install_path}")
    docker_root_install_path=$(change_install_path "${docker_root_install_path}")
    if [ ! -d "${docker_root_install_path}" ]; then
        log "ERROR" "ERR_NO:0x0003;ERR_DES:The $docker_root_install_path dose not exist, please retry a right path."
        exit_install_log 1
    fi

    if [ ! -r "${docker_root_install_path}" ] || [ ! -w "${docker_root_install_path}" ] || [ ! -x "${docker_root_install_path}" ]; then
        log "ERROR" "ERR_NO:0x0003;ERR_DES:The $docker_root_install_path permission is invalid, please retry a right path."
        exit_install_log 1
    fi
    install_path=$(concat_docker_install_path "${docker_root_install_path}" "${install_path_param}")
    operation_log_file=$(concat_docker_install_path "${docker_root_install_path}" "${operation_log_file}")
    log_file=$(concat_docker_install_path "${docker_root_install_path}" "${log_file}")
    log "INFO" "Install log for docker root dir: $log_file."
    log_dir=$(concat_docker_install_path "${docker_root_install_path}" "${log_dir}")
else
    install_path=${install_path_param}
fi

#######################################################
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
get_version_dir "pkg_version_dir" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ]; then
    default_dir="${install_path}/${pkg_version_dir}/${pkg_relative_path}"
    chmod_single_dir "${install_path}/${pkg_version_dir}/${PACKAGE_PATH}" 750 "dir" 2> /dev/null
else
    default_dir="${install_path}/${pkg_relative_path}"
    chmod_single_dir "${install_path}/${PACKAGE_PATH}" 750 "dir" 2> /dev/null
fi

create_log_folder
change_log_mode
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
        log "WARNING" "Aoe package doesn't contain features $featuremode, skip installation."
        exit 0
    fi
fi

# pre-check
if [ "${input_pre_check}" = y ]; then
    log "INFO" "Aoe do pre check started."
    pre_check
    if [ $? -ne 0 ]; then
        log "WARNING" "Aoe do pre check failed."
    else
        log "INFO" "Aoe do pre check finished."
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

check_install_for_all
create_default_install_dir_for_common_user
log_base_version
is_valid_path
input_install_path="${input_install_path_param}"

if [ "$full_install" = y ] || [ "$run_install" = y ] || [ "$devel_install" = y ]; then
    create_default_dir
fi

# 环境上是否已安装过run包
version2=$(get_version_installed "${default_dir}")
if [ "$version2""x" != "x" -a "$version2" != "none" ] || [ -f "${install_info}" ]; then
    # 卸载场景
    if [ "$uninstall" = y ]; then
        uninstall_run "uninstall" ${default_dir} ${pkg_is_multi_version} $uninstall
        save_user_files_to_log "$default_dir"
    # 升级场景
    elif [ "$upgrade" = y ]; then
        if [ -n "$pkg_version_dir" ]; then
            get_package_upgrade_version_dir "upgrade_version_dir" "$install_path" "aoe"
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
        version1=$(get_version_in_run_file)
        # 判断是否要覆盖式安装
        if [ "$is_quiet" = n ]; then
            if [ "${input_install_path}" = "" ]; then
                path_=${default_root_dir}
            elif [ "${input_install_path}" != "" ]; then
                path_=${install_path}
            fi
            log "INFO" "Aoe package has been installed on the path ${path_}, the version is ${version2}, and the version of this package is ${version1}, do you want to continue?  [y/n] "
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
            get_package_upgrade_version_dir "upgrade_version_dir" "$install_path" "aoe"
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
