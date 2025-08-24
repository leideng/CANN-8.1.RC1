#!/bin/bash
# Perform install/upgrade/uninstall for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

default_root_dir="/usr/local/Ascend"
default_normal_dir="${HOME}/Ascend"
username=$(id -un)
usergroup=$(id -gn)
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
version_compat_func_path="${curpath}/version_compatiable.inc"
common_func_v2_path="${curpath}/common_func_v2.inc"
version_cfg_path="${curpath}/version_cfg.inc"
compiler_func_path="${curpath}/compiler_func.sh"
pkg_version_path="${curpath}/../../version.info"
install_info_old="/etc/ascend_install.info"
run_dir="$(echo "$2" | cut -d'-' -f 3-)"

. "${common_func_path}"
. "${version_compat_func_path}"
. "${common_func_v2_path}"
. "${version_cfg_path}"
. "${compiler_func_path}"

if [ "$(id -u)" != "0" ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

operation_logfile="${log_dir}/operation.log"
logfile="${log_dir}/ascend_install.log"

# 递归授权
chmod_recur() {
    local file_path="${1}"
    local permission="${2}"
    local type="${3}"
    permission=$(set_file_chmod "$permission")
    if [ "$type" = "dir" ]; then
        find "$file_path" -type d -exec chmod "$permission" {} \; 2> /dev/null
    elif [ "$type" = "file" ]; then
        find "$file_path" -type f -exec chmod "$permission" {} \; 2> /dev/null
    fi
}

# 单目录授权
chmod_single_dir() {
    local file_path="${1}"
    local permission="${2}"
    local type="${3}"
    permission=$(set_file_chmod "$permission")
    if [ "$type" = "dir" ]; then
        chmod "$permission" "$file_path"
    elif [ "$type" = "file" ]; then
        chmod "$permission" "$file_path"
    fi
}

# 设置权限
set_file_chmod() {
    local permission="${1}"
    local new_permission=""
    if [ "${input_install_for_all}" = "y" ]; then
        new_permission="$(expr substr $permission 1 2)$(expr substr $permission 2 1)"
        echo "$new_permission"
    else
        echo "$permission"
    fi
}

# 运行前授权
chmod_start() {
    local tmpdir="$1"
    [ -z "$tmpdir" ] && tmpdir="$default_dir"
    chmod_recur "$tmpdir" 750 dir 2> /dev/null
    chmod_recur "$tmpdir/python" 750 dir 2> /dev/null
}

# 运行结束授权
chmod_end() {
    local current_install_path="$pkg_install_path"
    if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" != "y" ]; then
        current_install_path="$current_install_path/$pkg_version_dir"
    fi

    chmod_recur "$default_dir/ccec_compiler/lib" 440 file
    chmod_recur "$default_dir/ccec_compiler/include" 440 file
    # data dir/file permission
    chmod_recur "$default_dir/python" 750 dir
    chmod_recur "$current_install_path/python" 750 dir
    chmod_recur "$current_install_path/python/site-packages/auto_tune" 750 file
    chmod_recur "$current_install_path/python/site-packages/auto_tune-0.1.0.dist-info" 550 dir
    chmod_recur "$current_install_path/python/site-packages/auto_tune-0.1.0.dist-info" 550 file
    chmod_recur "$current_install_path/python/site-packages/auto_deploy_utils" 750 file
    chmod_recur "$current_install_path/python/site-packages/repository_manager" 750 file
    chmod_recur "$current_install_path/python/site-packages/tik_tune" 750 file
    chmod_recur "$current_install_path/python/site-packages/schedule_search" 750 file
    chmod_recur "$current_install_path/python/site-packages/schedule_search.egg" 750 file
    chmod_recur "$current_install_path/python/site-packages/schedule_search-0.0.1.dist-info" 550 dir
    chmod_recur "$current_install_path/python/site-packages/schedule_search-0.0.1.dist-info" 550 file
    chmod_recur "$current_install_path/python/site-packages/optune_utils" 750 file
    chmod_recur "$current_install_path/python/site-packages/opc_tool" 550 dir
    chmod_recur "$current_install_path/python/site-packages/opc_tool" 550 file
    chmod_recur "$current_install_path/python/site-packages/opc_tool-0.1.0.dist-info" 550 dir
    chmod_recur "$current_install_path/python/site-packages/opc_tool-0.1.0.dist-info" 550 file
    chmod_recur "$current_install_path/python/site-packages/op_compile_tool" 550 dir
    chmod_recur "$current_install_path/python/site-packages/op_compile_tool" 550 file
    chmod_recur "$current_install_path/python/site-packages/op_compile_tool-0.1.0.dist-info" 550 dir
    chmod_recur "$current_install_path/python/site-packages/op_compile_tool-0.1.0.dist-info" 550 file
    chmod_recur "$current_install_path/python/site-packages/autofuse" 550 dir
    chmod_recur "$current_install_path/python/site-packages/autofuse" 550 file
    chmod_recur "$current_install_path/../latest/python/site-packages" 750 dir

    if [ "$pylocal" = "y" ]; then
        chmod_recur "$current_install_path/python/site-packages/dataflow" 550 dir
        chmod_recur "$current_install_path/python/site-packages/dataflow" 550 file
        chmod_recur "$current_install_path/python/site-packages/dataflow-0.0.1.dist-info" 550 dir
        chmod_recur "$current_install_path/python/site-packages/dataflow-0.0.1.dist-info" 550 file
        chmod_recur "$current_install_path/python/site-packages/llm_datadist" 550 dir
        chmod_recur "$current_install_path/python/site-packages/llm_datadist" 550 file
        chmod_recur "$current_install_path/python/site-packages/llm_datadist-0.0.1.dist-info" 550 dir
        chmod_recur "$current_install_path/python/site-packages/llm_datadist-0.0.1.dist-info" 550 file
        chmod_recur "$current_install_path/python/site-packages/hccl" 550 dir
        chmod_recur "$current_install_path/python/site-packages/hccl" 550 file
        chmod_recur "$current_install_path/python/site-packages/hccl-0.1.0.dist-info" 550 dir
        chmod_recur "$current_install_path/python/site-packages/hccl-0.1.0.dist-info" 550 file
        chmod_recur "$current_install_path/python/site-packages/te" 550 dir
        chmod_recur "$current_install_path/python/site-packages/te" 550 file
        chmod_recur "$current_install_path/python/site-packages/te-0.4.0.dist-info" 550 dir
        chmod_recur "$current_install_path/python/site-packages/te-0.4.0.dist-info" 550 file
        chmod_recur "$current_install_path/python/site-packages/te_fusion" 750 file
        chmod_recur "$current_install_path/python/site-packages/tbe" 750 file
        chmod_recur "$current_install_path/python/site-packages/LICENSE" 440 file
    fi

    chmod_single_dir "$default_dir/ascend_install.info" 640 file 2> /dev/null
    chmod_single_dir "$default_dir/version.info" 440 file 2> /dev/null
    chmod_single_dir "$default_dir/scene.info" 640 file 2> /dev/null
    chmod_single_dir "$default_dir" 550 dir 2> /dev/null
    chmod_single_dir "$default_dir/script" 550 dir 2> /dev/null
    chmod_recur "$default_dir/script" 550 file 2> /dev/null
    chown -R "$username":"$usergroup" "$default_dir" 2> /dev/null
    if [ $(id -u) -eq 0 ]; then
        chown "root:root" "$default_dir" 2> /dev/null
        chmod 755 "$default_dir" 2> /dev/null
        chown -R "root:root" "$default_dir/script" 2> /dev/null
    fi
}

ver_check() {
    local version_info_file=""
    if [ -f "${install_info_old}" ] && [ $(grep -c -i driver_install_path_param "$install_info_old") -ne 0 ]; then
        if [ -f "${default_dir}/version.info" ]; then
            version_info_file="${default_dir}/version.info"
        else
            version_info_file="$pkg_version_path"
        fi
        local driver_install_path_param="$(grep -iw driver_install_path_param "${install_info_old}" | cut -d"=" -f2-)"
        local dep_pkg_ver_file="${driver_install_path_param}/driver/version.info"
        if [ "$check" = "y" ]; then
            bash "${curpath}/ver_check.sh" "${version_info_file}" "driver" "${dep_pkg_ver_file}"
        elif [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ] || [ "$upgrade" = "y" ]; then
            bash "${curpath}/ver_check.sh" "${version_info_file}" "driver" "${dep_pkg_ver_file}" 1> /dev/null
            local ret=$?
            if [ "${ret}" -eq 1 ] && [ "$is_quiet" = "n" ]; then
                log "WARNING" "Check version does not matched, do you want to continue? [y/n]"
                while true
                do
                    read yn
                    if [ "$yn" = "n" ]; then
                        log "INFO" "stop installation!"
                        exit_install_log 0
                    elif [ "$yn" = "y" ]; then
                        break
                    else
                        log "ERROR" "ERR_NO:0x0002;ERR_DES:input error, please input again!"
                    fi
                done
            elif [ "${ret}" -eq 1 ] && [ "$is_quiet" = "y" ]; then
                log "WARNING" "Check version does not matched!"
            elif [ "${ret}" -eq 0 ]; then
                log "INFO" "Check version matched!"
            fi
        fi
    else
        log "WARNING" "Cannot find the install path of driver."
    fi
}

param_usage() {
    log "INFO" "Please input this command for help: ./${runfilename} --help"
}

# 修改日志文件的权限
change_log_mode() {
    if [ ! -f "$logfile" ]; then
        touch "$logfile"
    fi
    chmod 640 "$logfile"
}

# 创建日志文件夹
create_log_folder() {
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi
    if [ $(id -u) -ne 0 ]; then
        chmod 740 "$log_dir"
    else
        chmod 750 "$log_dir"
    fi
}

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    local log_msg="$2"
    local log_format="[Compiler] [$cur_date] [$log_type]: $log_msg"
    if [ "$log_type" = "INFO" ]; then
        echo "$log_format"
    elif [ "$log_type" = "WARNING" ]; then
        echo "$log_format"
    elif [ "$log_type" = "ERROR" ]; then
        echo "$log_format"
    elif [ "$log_type" = "DEBUG" ]; then
        echo "$log_format" 1> /dev/null
    fi
    if [ -d "$log_dir" ]; then
        echo "$log_format" >> "$logfile"
    fi
}

# 静默模式日志打印
new_echo() {
    local log_type="$1"
    local log_msg="$2"
    if [ "${is_quiet}" = "n" ]; then
        echo "${log_type}" "${log_msg}" 1> /dev/null
    fi
}

# 开始安装前打印开始信息
start_install_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    new_echo "INFO" "Start time:${cur_date}"
    log "INFO" "Start time:$cur_date"
    log "INFO" "LogFile:${logfile}"
    log "INFO" "InputParams:$all_parma"
    log "INFO" "OperationLogFile:${operation_logfile}"
}

# 开始卸载前打印开始信息
start_uninstall_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    new_echo "INFO" "Start time:${cur_date}"
    log "INFO" "Start time:$cur_date"
    log "INFO" "LogFile:${logfile}"
    log "INFO" "InputParams:$all_parma"
    log "INFO" "OperationLogFile:${operation_logfile}"
}

# 安装结束退出前打印结束信息
exit_install_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    new_echo "INFO" "End time:${cur_date}"
    log "INFO" "End time:${cur_date}"
    exit "$1"
}

# 安装结束退出前打印结束信息
exit_uninstall_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    new_echo "INFO" "End time:${cur_date}"
    log "INFO" "End time:${cur_date}"
    exit "$1"
}

# 安全日志
log_operation() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local level=""
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
        touch "${operation_logfile}"
        chmod 640 "${operation_logfile}"
    fi

    echo "$1 $level root $cur_date 127.0.0.1 $runfilename $2 installmode=$installmode; cmdlist=$all_parma" >> "$operation_logfile"
}

# 相对路径转化绝对路径
relative_path_to_absolute_path() {
    local relative_path_="${1}"
    if [ "x$relative_path_" = "x" ]; then
        return
    fi
    local fstr="$(expr substr "$relative_path_" 1 1)"
    if [ "$fstr" = "~" ]; then
        relative_path_="${HOME}$(echo "${relative_path_}" | cut -d'~' -f 2-)"
    elif [ "$fstr" != "/" ]; then
        relative_path_="${run_dir}/${relative_path_}"
    fi
    echo "$relative_path_"
}

# 获取安装路径
get_install_path() {
    local ppath=""
    if [ "$input_path_flag" = "y" ]; then
        if [ "x${input_install_path}" = "x" ]; then
            log "ERROR" "ERR_NO:0x0004;ERR_DES: Install path is empty."
            exit 1
        fi
    fi
    input_install_path=$(echo "${input_install_path}" | sed "s/\/*$//g")
    if [ "x${input_install_path}" = "x" ]; then
        input_install_path="/"
    fi
    if [ "$uninstall" != "y" ]; then
        ppath=$(echo "${input_install_path}" | sed "s/\/[^/]*$//g")
        if [ "x${ppath}" != "x" ] && [ ! -d "${ppath}" ]; then
            log "ERROR" "parent path doesn't exist, please create ${ppath} first."
            exit 1
        fi
    fi
}

create_install_dir() {
    local path="$1"
    local user_and_group="$2"
    local permission=""

    if [ $(id -u) -eq 0 ]; then
        user_and_group="root:root"
        permission=755
    else
        if [ "$input_install_for_all" = "y" ]; then
            permission=755
        else
            permission=750
        fi
    fi

    if [ "x${path}" = "x" ]; then
        log "WARNING" "dir path is empty"
        return 1
    fi
    mkdir -p "${path}"
    if [ $? -ne 0 ]; then
        log "WARNING" "create path=${path} failed."
        return 1
    fi
    chmod "$permission" "${path}"
    if [ $? -ne 0 ]; then
        log "WARNING" "chmod path=${path} $permission failed."
        return 1
    fi
    chown -f "$user_and_group" "${path}"
    if [ $? -ne 0 ]; then
        log "WARNING" "chown path=${path} $user_and_group failed."
        return 1
    fi
}

create_file() {
    local _file="$1"
    if [ -d "${_file}" ]; then
        rm -rf "${_file}"
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
    if [ "x${pkg_install_path}" != "x" ]; then
        if [ ! -d "${pkg_install_path}" ]; then
            local up_dir=$(dirname "${pkg_install_path}")
            if [ ! -d "${up_dir}" ]; then
                log "ERROR" "ERR_NO:0x0003;ERR_DES:The $up_dir dose not exist, please retry a right path."
                exit_install_log 1
            fi
        else
            local install_path="$pkg_install_path"
            if [ "$hetero_arch" = "y" ]; then
                install_path="$(realpath $install_path/../..)"
            fi
            local ret=0
            if [ $(id -u) -eq 0 ]; then
                parent_dirs_permission_check "$install_path" && ret=$? || ret=$?
                if [ "${is_quiet}" = "y" ] && [ "${ret}" -ne 0 ]; then
                    log "ERROR" "the given dir, or its parents, permission is invalid."
                    exit 1
                fi
                if [ "${ret}" -ne 0 ]; then
                    log "WARNING" "You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
                    while true
                    do
                        read yn
                        if [ "$yn" = "n" ]; then
                            exit 1
                        elif [ "$yn" = "y" ]; then
                            break
                        else
                            echo "ERR_NO:0x0002;ERR_DES:input error, please input again!"
                        fi
                    done
                fi
            fi
            if [ $(id -u) -ne 0 ]; then
                log "DEBUG" "$install_path"
            else
                sh -c 'cd "$install_path" >> /dev/null 2>&1'
            fi
            if [ $? != 0 ]; then
                log "ERROR" "ERR_NO:0x0093;ERR_DES:The $username do not have the permission to access $install_path, please reset the directory to a right permission."
                exit_install_log 1
            fi
        fi
    fi
}

# 递归判断安装路的属组和权限
parent_dirs_permission_check() {
    local current_dir="$1"
    local parent_dir=""
    local short_install_dir=""
    local owner=""
    local mod_num=""

    parent_dir=$(dirname "${current_dir}")
    short_install_dir=$(basename "${current_dir}")
    log "INFO" "parent_dir value is [${parent_dir}] and children_dir value is [${short_install_dir}]"

    if [ "x${current_dir}" = "x/" ]; then
        log "INFO" "parent_dirs_permission_check succeeded"
        return 0
    else
        owner=$(stat -c %U "${parent_dir}/${short_install_dir}")
        if [ "${owner}" != "root" ]; then
            log "WARNING" "[${short_install_dir}] permission isn't right, it should belong to root."
            return 1
        fi
        log "INFO" "[${short_install_dir}] belongs to root."

        mod_num=$(stat -c %a "${parent_dir}/${short_install_dir}")
        mod_num=$(check_chmod_length "$mod_num")
        if [ "${mod_num}" -lt 755 ]; then
            log "WARNING" "[${short_install_dir}] permission is too small, it is recommended that the permission be 755 for the root user."
            return 2
        elif [ "${mod_num}" -eq 755 ]; then
            log "INFO" "[${short_install_dir}] permission is ok."
        else
            log "WARNING" "[${short_install_dir}] permission is too high, it is recommended that the permission be 755 for the root user."
            [ "${is_quiet}" = "n" ] && return 3
        fi
        parent_dirs_permission_check "${parent_dir}"
    fi
}

check_chmod_length() {
    local mod_num="$1"
    local new_mod_num=""
    local mod_num_length=$(expr length "$mod_num")
    if [ "$mod_num_length" -eq 3 ]; then
        new_mod_num="$mod_num"
        echo "$new_mod_num"
    elif [ "$mod_num_length" -eq 4 ]; then
        new_mod_num="$(expr substr $mod_num 2 3)"
        echo "$new_mod_num"
    fi
}

# 校验用户和组的关联关系
check_group() {
    local group_user_related=""
    local result="$(groups "$2" | grep ":")"
    if [ "x${result}" != "x" ]; then
        group_user_related=$(groups "$2"|awk -F":" '{print $2}'|grep -w "$1")
    else
        group_user_related=$(groups "$2"|grep -w "$1")
    fi
    if [ "x${group_user_related}" != "x" ]; then
        return 0
    else
        return 1
    fi
}

# 解锁
unchattr_files() {
    if [ -f "$install_info" ]; then
        if [ -d "$default_dir" ]; then
            chattr -R -i "$default_dir" >> /dev/null 2>&1
            if [ $? -ne 0 ]; then
                log "DEBUG" "unchattr -i for the subfiles."
                find "$default_dir" -name "*" | xargs chattr -i  >> /dev/null 2>&1
            else
                log "DEBUG" "unchattr -R -i $pkg_install_path succeeded."
            fi
        fi
    fi
}

# 创建普通用户的默认安装目录
create_default_install_dir_for_common_user() {
    if [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ]; then
        if [ ! -d "$pkg_install_path" ]; then
            create_install_dir "$pkg_install_path" "$username:$usergroup"
        fi
        if [ "$hetero_arch" = "y" ]; then
            local parent_dir="$(dirname "$pkg_install_path")"
            create_install_dir "$parent_dir" "$username:$usergroup"
        fi
    fi
}

# 创建子包安装目录
create_default_dir() {
    if [ ! -d "$default_dir" ]; then
        create_install_dir "$default_dir" "${username}":"${usergroup}"
    fi
    if [ -n "$pkg_version_dir" ] && [ "$hetero_arch" != "y" ]; then
        create_install_dir "$(dirname $default_dir)" "$username:$usergroup"
    fi
    [ -d "$default_dir" ] && return 0
    return 1
}

# 获取安装目录下的完整版本号
get_version_installed() {
    local installed_version="none"
    if [ -f "$default_dir/version.info" ]; then
        installed_version="$(grep -iw ^version "$default_dir/version.info" | cut -d"=" -f2-)"
    fi
    echo "$installed_version"
}

# 获取本包中的完整版本号
get_version_in_runpkg() {
    local version_in_runpkg="none"
    if [ -f "$pkg_version_path" ]; then
        version_in_runpkg="$(grep -iw ^version $pkg_version_path | cut -d"=" -f2-)"
    fi
    echo "$version_in_runpkg"
}

# 更新基础版本号
update_version_info_version() {
    if [ -f "$default_dir/version.info" ]; then
        rm -f "$default_dir/version.info"
        cp -f "$pkg_version_path" "$default_dir"
        log "INFO" "Upgrade base version successfully!"
    else
        cp -f "$pkg_version_path" "$default_dir"
        log "INFO" "Base version set successfully!"
    fi
    chmod_single_dir "$default_dir/version.info" 440 file >> /dev/null 2>&1
}

log_base_version() {
    if [ -f "$install_info" ]; then
        local installed_version="$(get_version_installed)"
        if [ "x${installed_version}" != "x" ]; then
            new_echo "INFO" "base version is ${installed_version}."
            log "INFO" "base version is ${installed_version}."
            return 0
        fi
    fi
    if [ "$upgrade" = "y" ]; then
        new_echo "WARNING" "base version was destroyed or not exist."
        log "WARNING" "base version was destroyed or not exist."
    fi
}

update_install_path() {
    if [ ! -d "$pkg_install_path" ]; then
        log "ERROR" "ERR_NO:0x0003;ERR_DES:The $pkg_install_path dose not exist, please retry a right path."
        exit_install_log 1
    fi
}

update_install_param() {
    local _key="$1"
    local _val="$2"
    local _file="$3"
    local _param=""
    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    local install_info_key_array="Compiler_Install_Type Compiler_Chip_Type Compiler_Feature_Type Compiler_UserName Compiler_UserGroup Compiler_Install_Path_Param Compiler_Arch_Linux_Path Compiler_Hetero_Arch_Flag"
    for key_param in ${install_info_key_array}; do
        if [ "${key_param}" = "${_key}" ]; then
            _param=$(grep -i "${_key}=" "${_file}")
            if [ "x${_param}" = "x" ]; then
                echo "${_key}=${_val}" >> "${_file}"
            else
                sed -i "/^${_key}=/Ic ${_key}=${_val}" "${_file}" 2> /dev/null
            fi
            break
        fi
    done
}

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param=""

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    local install_info_key_array="Compiler_Install_Type Compiler_Chip_Type Compiler_Feature_Type Compiler_UserName Compiler_UserGroup Compiler_Install_Path_Param Compiler_Arch_Linux_Path Compiler_Hetero_Arch_Flag"
    for key_param in ${install_info_key_array}; do
        if [ "${key_param}" = "${_key}" ]; then
            _param=$(grep -i "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

update_install_info_feature() {
    local operation="$1"
    if [ "$featuremode" = "all" ] || [ "$operation" = "Upgrade" ]; then
        update_install_param "Compiler_Feature_Type" "$featuremode" "$install_info"
        return
    fi
    local current_featuremode=$(get_install_param "Compiler_Feature_Type" "$install_info")
    if [ -z "$current_featuremode" ] || [ "$current_featuremode" = "all" ]; then
        update_install_param "Compiler_Feature_Type" "$featuremode" "$install_info"
        return
    fi
    local version_in_runpkg="$(get_version_in_runpkg)"
    if [ "$version_in_runpkg" != "$version_installed" ]; then
        update_install_param "Compiler_Feature_Type" "$featuremode" "$install_info"
        return
    fi
    featuremode=$(echo "$current_featuremode,$featuremode" | tr ',' '\n' | sort -u | tr '\n' ',' | sed -e 's/^,\+\|,\+$//g')
    update_install_param "Compiler_Feature_Type" "$featuremode" "$install_info"
}

update_install_info() {
    chmod_start
    if [ ! -f "$install_info" ]; then
        create_file "$install_info" "$username":"$usergroup" 640
    fi
    update_install_param "Compiler_Install_Type" "$installmode" "$install_info"
    update_install_param "Compiler_Chip_Type" "$chipmode" "$install_info"
    update_install_info_feature
    update_install_param "Compiler_UserName" "$username" "$install_info"
    update_install_param "Compiler_UserGroup" "$usergroup" "$install_info"
    update_install_param "Compiler_Install_Path_Param" "$install_path_param" "$install_info"
    update_install_param "Compiler_Arch_Linux_Path" "$arch_linux_path" "$install_info"
    update_install_param "Compiler_Hetero_Arch_Flag" "$hetero_arch" "$install_info"
}

prompt_set_env() {
    local install_path="$1"
    if [ -n "$pkg_version_dir" ] && [ "$hetero_arch" != "y" ]; then
        install_path="$install_path/$pkg_version_dir"
    fi
    if [ "$hetero_arch" = "y" ]; then
        echo "Please make sure that
            - PATH includes ${install_path}/compiler/ccec_compiler/bin:${install_path}/compiler/bin
            - LD_LIBRARY_PATH includes ${install_path}/compiler/lib64:${install_path}/compiler/lib64/plugin/opskernel:${install_path}/compiler/lib64/plugin/nnengine"
    else
        echo "Please make sure that
            - PATH includes ${install_path}/compiler/ccec_compiler/bin:${install_path}/compiler/bin
            - LD_LIBRARY_PATH includes ${install_path}/compiler/lib64:${install_path}/compiler/lib64/plugin/opskernel:${install_path}/compiler/lib64/plugin/nnengine
            - PYTHONPATH includes ${install_path}/compiler/python/site-packages"
    fi
}

check_docker_path() {
    local docker_path="$1"
    if [ $(expr substr "${docker_path}" 1 1) != "/" ]; then
        log "ERROR" "ERR_NO:0x0002;ERR_DES:Parameter --docker-root \
        must with absolute path that which is start with root directory /. Such as --docker-root=/${docker_path}"
        exit 1
    fi
    if [ ! -d "${docker_path}" ]; then
        log "ERROR" "ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist, please create this directory."
        exit 1
    fi
}

concat_docker_install_path() {
    local docker_path="$1"
    local install_path=""
    docker_path=$(echo "${docker_path}" | sed "s/\/*$//g")
    if [ "x${docker_path}" = "x" ]; then
        docker_path="/"
    fi
    install_path="${docker_path}$2"
    echo "${install_path}"
}

#######################################################
# 安装调用子脚本
install_run() {
    local operation="Install"
    local compiler_install_path_param=""
    update_install_path
    update_install_info
    local compiler_input_install_path=$(get_install_param "Compiler_Install_Path_Param" "${install_info}")
    local compiler_install_type=$(get_install_param "Compiler_Install_Type" "${install_info}")
    if [ "$is_docker_install" = "y" ]; then
        compiler_install_path_param=$(concat_docker_install_path "${docker_root}" "${compiler_input_install_path}")
    else
        compiler_install_path_param="${compiler_input_install_path}"
    fi
    if [ "$1" = "install" ]; then
        unchattr_files
        chmod_start
        new_echo "INFO" "install ${compiler_install_path_param} ${compiler_install_type}"
        log "INFO" "install ${compiler_install_path_param} ${compiler_install_type}"
        bash "${curpath}/run_compiler_install.sh" "install" "${compiler_input_install_path}" "${compiler_install_type}" \
            "${is_quiet}" "${pylocal}" "${input_setenv}" "${docker_root}" "${in_install_for_all}"
        if [ $? -eq 0 ]; then
            update_version_info_version
            if [ "$hetero_arch" = "y" ]; then
                create_atc_fwkacllib_softlink "${compiler_install_path_param}" "${in_install_for_all}"
            else
                create_atc_fwkacllib_softlink "${compiler_install_path_param}/${pkg_version_dir}" "${in_install_for_all}"
            fi
            log "INFO" "Compiler package installed successfully! The new version takes effect immediately."
            log_operation "${operation}" "succeeded"
            chmod_end
            prompt_set_env "${install_path_param}"
        else
            chmod_end
            log "ERROR" "Compiler package install failed, please retry after uninstall!"
            log_operation "${operation}" "failed!"
            exit_install_log 1
        fi
    fi
    return $?
}

# 升级调用子脚本
upgrade_run() {
    local operation="Upgrade"
    local compiler_install_path_param=""
    update_install_info_feature "$operation"
    if [ -f "$install_info" ]; then
        local compiler_input_install_path=$(get_install_param "Compiler_Install_Path_Param" "${install_info}")
        local compiler_install_type=$(get_install_param "Compiler_Install_Type" "${install_info}")
    elif [ -f "$install_info_old" ] && [ $(grep -c -i compiler_install_path_param "$install_info_old") -ne 0 ]; then
        local compiler_input_install_path="$(grep -iw compiler_install_path_param "$install_info_old" | cut -d"=" -f2-)"
        local compiler_install_type="$(grep -iw compiler_install_type "$install_info_old" | cut -d"=" -f2-)"
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:Installation information no longer exists, please complete ${install_info} or ${install_info_old}"
        exit_install_log 1
    fi
    if [ "$is_docker_install" = "y" ]; then
        compiler_install_path_param=$(concat_docker_install_path "${docker_root}" "${compiler_input_install_path}")
    else
        compiler_install_path_param="${compiler_input_install_path}"
    fi
    if [ "$1" = "upgrade" ]; then
        chmod_start
        new_echo "INFO" "upgrade ${compiler_install_path_param} ${compiler_install_type}"
        log "INFO" "upgrade ${compiler_install_path_param} ${compiler_install_type}"
        bash "${curpath}/run_compiler_upgrade.sh" "upgrade" "${compiler_input_install_path}" "${compiler_install_type}" \
            "${is_quiet}" "${pylocal}" "${input_setenv}" "${docker_root}" "${in_install_for_all}"
        if [ $? -eq 0 ]; then
            update_version_info_version
            if [ "$hetero_arch" = "y" ]; then
                create_atc_fwkacllib_softlink "${compiler_install_path_param}" "${in_install_for_all}"
            else
                create_atc_fwkacllib_softlink "${compiler_install_path_param}/${pkg_version_dir}" "${in_install_for_all}"
            fi
            log "INFO" "Compiler package upgraded successfully! The new version takes effect immediately."
            log_operation "${operation}" "succeeded"
            chmod_end
            prompt_set_env "${install_path_param}"
        else
            chmod_end
            log "ERROR" "Compiler package upgrade failed, please retry after uninstall!"
            log_operation "${operation}" "failed!"
            exit_install_log 1
        fi
    fi
    return $?
}

# 卸载调用子脚本
uninstall_run() {
    local is_recreate_softlink="$2"
    local is_remove_info_files="$3"
    local upgrade_install_info="$4"
    [ -z "$upgrade_install_info" ] && upgrade_install_info="$install_info"
    local operation="Uninstall"
    local compiler_install_path_param=""
    if [ -f "$upgrade_install_info" ]; then
        local compiler_input_install_path=$(get_install_param "Compiler_Install_Path_Param" "${upgrade_install_info}")
        local compiler_install_type=$(get_install_param "Compiler_Install_Type" "${upgrade_install_info}")
    elif [ -f "$install_info_old" ] && [ $(grep -c -i compiler_install_path_param "$install_info_old") -ne 0 ]; then
        local compiler_input_install_path="$(grep -iw compiler_install_path_param "$install_info_old" | cut -d"=" -f2-)"
        local compiler_install_type="$(grep -iw compiler_install_type "$install_info_old" | cut -d"=" -f2-)"
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:Installation information no longer exists, please complete ${upgrade_install_info} or ${install_info_old}"
        exit_uninstall_log 1
    fi
    if [ "$is_docker_install" = "y" ]; then
        compiler_install_path_param=$(concat_docker_install_path "${docker_root}" "${compiler_input_install_path}")
    else
        compiler_install_path_param="${compiler_input_install_path}"
    fi
    local upgrade_default_dir="$(dirname $upgrade_install_info)"
    if [ ! -f "$upgrade_default_dir/script/run_compiler_uninstall.sh" ]; then
        log "WARNING" "run_compiler_uninstall.sh not found."
        return $?
    fi
    if [ "$1" = "uninstall" ]; then
        chmod_start "$upgrade_default_dir"
        new_echo "INFO" "uninstall ${compiler_install_path_param} ${compiler_install_type}"
        log "INFO" "uninstall ${compiler_install_path_param} ${compiler_install_type}"
        bash "$upgrade_default_dir/script/run_compiler_uninstall.sh" "uninstall" "${compiler_input_install_path}" "${compiler_install_type}" "${is_quiet}" \
            "${is_docker_install}" "${docker_root}" "${is_recreate_softlink}"
        if [ $? -eq 0 ]; then
            if [ "$is_remove_info_files" = "y" ]; then
                test -f "$upgrade_install_info" && rm -f "$upgrade_install_info"
                test -f "$upgrade_default_dir/version.info" && rm -f "$upgrade_default_dir/version.info"
            fi
            if [ "$hetero_arch" = "y" ]; then
                remove_atc_fwkacllib_softlink "${compiler_install_path_param}"
            else
                local version_dir="$([ -n "$pkg_version_dir" ] && basename $(dirname $upgrade_default_dir))"
                remove_atc_fwkacllib_softlink "${compiler_install_path_param}/${version_dir}"
            fi
            remove_dir_recursive "$install_top_path" "$upgrade_default_dir"
            if [ $(id -u) -eq 0 ]; then
                if [ "$uninstall" = "y" ] && [ -f "$install_info_old" ] && [ $(grep -c -i compiler_install_path_param "$install_info_old") -ne 0 ]; then
                    sed -i '/compiler_install_path_param=/Id' "$install_info_old" 2> /dev/null
                    sed -i '/compiler_install_type=/Id' "$install_info_old" 2> /dev/null
                fi
            fi
            new_echo "INFO" "Compiler package uninstalled successfully! Uninstallation takes effect immediately."
            log "INFO" "Compiler package uninstalled successfully! Uninstallation takes effect immediately."
            log_operation "${operation}" "succeeded"
        else
            log "ERROR" "Compiler package uninstall failed!"
            log_operation "${operation}" "failed!"
            exit_uninstall_log 1
        fi
    fi
    return $?
}

save_user_files_to_log() {
    if [ "$1" = "${default_dir}" ] && [ -s "$1" ]; then
        local filenum=$(ls -lR "$1"|grep "^-"|wc -l)
        local dirnum=$(ls -lR "$1"|grep "^d"|wc -l)
        local totalnum=$(expr "${filenum}" + "${dirnum}")
        if [ "$totalnum" -eq 2 ]; then
            if [ -f "${install_info}" ] && [ -f "${default_dir}/version.info" ]; then
                return 0
            fi
        fi
        if [ "$totalnum" -eq 1 ]; then
            if [ -f "${install_info}" ] || [ -f "${default_dir}/version.info" ]; then
                return 0
            fi
        fi
        log "INFO" "Some files generated by user are not cleared, if necessary, manually clear them, get details in $logfile"
    fi
    if [ -s "$1" ]; then
        for file in $(ls -a "$1"); do
            if [ -d "$1/$file" -a ! -L "$1/$file" ]; then
                if [ "$file" != '.' ] && [ "$file" != '..' ]; then
                    echo "$1/$file" >> "$logfile"
                    save_user_files_to_log "$1/$file"
                fi
            else
                echo "$1/$file" >> "$logfile"
            fi
        done
    fi
}

judgmentpath() {
    . "${common_func_path}"
    check_install_path_valid "${1}"
    if [ $? -ne 0 ]; then
        log "ERROR" "The Compiler install_path ${1} is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
        exit 1
    fi
}

unique_mode() {
    if [ ! -z "$g_param_check_flag" ]; then
        log "ERROR" "ERR_NO:0x0004;ERR_DES:only support one type: full/run/docker/devel/upgrade/uninstall, operation failed!"
        exit 1
    fi
}

check_install_for_all() {
    local mod_num=""
    local other_mod_num=""
    if [ "$input_install_for_all" = "y" ] && [ -d "$pkg_install_path" ]; then
        mod_num="$(stat -c %a ${pkg_install_path})"
        mod_num="$(check_chmod_length $mod_num)"
        other_mod_num="$(expr substr $mod_num 3 1)"
        if [ "${other_mod_num}" -ne 5 ] && [ "${other_mod_num}" -ne 7 ]; then
            log "ERROR" "${pkg_install_path} permission is ${mod_num}, this permission does not support install_for_all param."
            exit_install_log 1
        fi
    fi
}

pre_check() {
    local check_shell_path="${curpath}/../bin/prereq_check.bash"
    if [ ! -f "${check_shell_path}" ]; then
        log "WARNING" "${check_shell_path} not exist."
        return 0
    fi
    if [ "x$is_quiet" = "xy" ]; then
        bash "${check_shell_path}" --quiet
    else
        bash "${check_shell_path}" --no-quiet
    fi
}

uninstall_none_multi_version() {
    if [ "$hetero_arch" = "y" ]; then
        return
    fi
    local install_path="$1"
    if [ "$pkg_is_multi_version" = "true" ] && [ ! -L "$install_path" ] && [ -d "$install_path" ]; then
        if [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ] || [ "$upgrade" = "y" ] || [ "$uninstall" = "y" ]; then
            local path_version="$install_path/version.info"
            local path_install="$install_path/ascend_install.info"
            if [ -f "$path_version" ] && [ -f "$path_install" ] && [ -f "$install_path/script/uninstall.sh" ]; then
                $install_path/script/uninstall.sh
            fi
        fi
    fi
}

migrate_user_assets_v2() {
    if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" != "y" ]; then
        get_package_last_installed_version_dir "last_installed_dir" "$pkg_install_path" "compiler"
        if [ -n "$last_installed_dir" ]; then
            last_installed_dir="$pkg_install_path/$last_installed_dir"
            local current_installed_dir="${pkg_install_path}/$pkg_version_dir"
            local data_dir="compiler/data/fusion_strategy"
            if [ -d "$last_installed_dir/$data_dir/custom" ] && [ "x$(ls -A $last_installed_dir/$data_dir/custom)" != "x" ]; then
                mkdir -p "$current_installed_dir/$data_dir"
                log "INFO" "cp -rpf $last_installed_dir/$data_dir/custom $current_installed_dir/$data_dir"
                cp -rpf "$last_installed_dir/$data_dir/custom" "$current_installed_dir/$data_dir"
            fi
        fi
    fi
}

####################################################################################################
runfilename=$(expr substr "$1" 5 $(expr ${#1} - 4))

full_install=n
run_install=n
docker_install=n
devel_install=n
uninstall=n
upgrade=n
installmode=""
chip_flag=n
chipmode="all"
feature_flag=n
featuremode="all"
pylocal=n
install_path_cmd="--install-path"
input_install_path=""
install_top_path=""
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
is_quiet=n
check=n
install_path_param=""
g_param_check_flag=""
hetero_arch=n

if [ $(id -u) -eq 0 ]; then
    input_install_for_all=y
    input_install_path="$default_root_dir"
    install_path_param="$default_root_dir"
    in_install_for_all="--install_for_all"
else
    input_install_path="$default_normal_dir"
    install_path_param="$default_normal_dir"
fi

create_log_folder
change_log_mode

####################################################################################################
if [ "$#" = "1" ] || [ "$#" = "2" ]; then
    log "ERROR" "ERR_NO:0x0004;ERR_DES:Unrecognized parameters. Try './xxx.run --help for more information.'"
    exit 1
fi

i=0
while true
do
    if [ "x$1" = "x" ]; then
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

all_parma="$*"

#################################################################################
while true
do
    case "$1" in
    --help | -h)
        param_usage
        exit 0
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
    --devel)
        unique_mode
        g_param_check_flag="True"
        devel_install=y
        installmode="devel"
        shift
        ;;
    --install-path=*)
        temp_path=$(echo "$1" | cut -d"=" -f2-)
        judgmentpath "${temp_path}"
        slashes_num=$(echo "${temp_path}" | grep -o '/' | wc -l)
        if [ "$slashes_num" -gt 1 ]; then
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
        temp_path=$(echo "$1" | cut -d"=" -f2-)
        judgmentpath "${temp_path}"
        slashes_num=$(echo "${temp_path}" | grep -o '/' | wc -l)
        if [ "$slashes_num" -gt 1 ]; then
            docker_root=$(echo "${temp_path}" | sed "s/\/*$//g")
        else
            docker_root="${temp_path}"
        fi
        is_docker_install=y
        check_docker_path "${docker_root}"
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
    --pylocal)
        pylocal=y
        shift
        ;;
    --extract=*)
        shift;
        ;;
    --keep)
        shift;
        ;;
    --check)
        check=y
        shift
        ;;
    --version)
        get_version_in_runpkg
        version=y
        exit 0
        ;;
    -*)
        log "ERROR" "ERR_NO:0x0004;ERR_DES: Unsupported parameters : $1"
        param_usage
        exit 0
        ;;
    *)
        break
        ;;
    esac
done

###################### 检查参数冲突 ###################
if [ "${is_quiet}" = "y" ]; then
    if [ "${upgrade}" = "y" ] || [ "${full_install}" = "y" ] || [ "${run_install}" = "y" ] || [ "${devel_install}" = "y" ] || [ "${uninstall}" = "y" ]; then
        is_quiet=y
    elif [ "${input_pre_check}" = "y" ] || [ "${check}" = "y" ]; then
        is_quiet=y
    else
        log "ERROR" "'--quiet' is not supported to used by this way, please use with '--full', '--devel', '--run', '--upgrade', '--uninstall', '--check' or '--pre-check'"
        exit 1
    fi
fi

# 检查chip参数是否冲突
if [ "${chip_flag}" = "y" ]; then
    if [ "$upgrade" = "n" ] && [ "$full_install" = "n" ] && [ "$run_install" = "n" ] && [ "$devel_install" = "n" ]; then
        log "ERROR" "ERR_NO:0x0004;ERR_DES: Operation failed, '--chip' must be configured together with '--run', '--devel', '--full' or '--upgrade'."
        exit 1
    fi
fi

# 检查feature参数是否冲突
if [ "${feature_flag}" = "y" ]; then
    if [ "$upgrade" = "n" ] && [ "$full_install" = "n" ] && [ "$run_install" = "n" ] && [ "$devel_install" = "n" ]; then
        log "ERROR" "ERR_NO:0x0004;ERR_DES: Operation failed, '--feature' must be configured together with '--run', '--devel', '--full' or '--upgrade'."
        exit 1
    fi
fi

if [ "${pylocal}" = "y" ]; then
    if [ "${upgrade}" != "y" ] && [ "${full_install}" != "y" ] && [ "${run_install}" != "y" ] && [ "${devel_install}" != "y" ]; then
        log "ERROR" "'--pylocal' is not supported to used by this way, please use with '--full', '--devel', '--run' or '--upgrade'."
        exit 1
    fi
fi

# 卸载参数只支持单独使用
if [ "$uninstall" = "y" ]; then
    if [ "$upgrade" = "y" ] || [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ] || [ "$docker_install" = "y" ] || [ "$check" = "y" ] || [ "$input_pre_check" = "y" ]; then
        log "ERROR" "ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
        exit 1
    fi
fi

if [ "$docker_install" = "y" ]; then
    log "ERROR" "ERR_NO:0x0004;ERR_DES:Unsupported parameters, operation failed."
    log "INFO" "--docker not uesd in compiler"
    exit 1
fi

# 检查必选参数
if [ "${upgrade}" = "n" ] && [ "${full_install}" = "n" ] && [ "${run_install}" = "n" ] && [ "${devel_install}" = "n" ] && [ "${uninstall}" = "n" ] && [ "${input_pre_check}" = "n" ] && [ "${check}" = "n" ]; then
    log "ERROR" "ERR_NO:0x0004;One of parameters '--full', '--devel', '--run', '--upgrade', '--uninstall', '--check' or '--pre-check' must be used."
    exit 1
fi

if [ "$featuremode" != "all" ]; then
    contain_feature "ret" "$featuremode" "$curpath/filelist.csv"
    if [ "$ret" = "false" ]; then
        log "WARNING" "Compiler package doesn't contain features $featuremode, skip installation."
        exit 0
    fi
fi

#######################################################
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
get_version_dir "pkg_version_dir" "$pkg_version_path"

if [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ] || [ "$upgrade" = "y" ] || [ "$uninstall" = "y" ]; then
    input_install_path=$(relative_path_to_absolute_path "${input_install_path}")
    get_install_path

    if [ "$(is_same_arch_pkg_installed)" = "y" ]; then
        hetero_arch="y"
    fi
    if [ "$upgrade" = "y" ] || [ "$uninstall" = "y" ]; then
        hetero_arch_pkg_installed="$(is_hetero_arch_pkg_installed)"
        if [ "$hetero_arch_pkg_installed" = "installed-hetero" ]; then
            hetero_arch="y"
        elif [ "$hetero_arch_pkg_installed" = "installed-normal" ]; then
            hetero_arch="n"
        elif [ "$hetero_arch_pkg_installed" = "installed-hetero-to-be-upgraded" ]; then
            hetero_arch="y"
        elif [ "$hetero_arch_pkg_installed" = "installed-normal-to-be-upgraded" ]; then
            hetero_arch="n"
        elif [ "$hetero_arch_pkg_installed" = "no" ]; then
            log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed in ${input_install_path}, operation failed!"
            exit 1
        fi
    fi
    export hetero_arch

    install_top_path="$(dirname $input_install_path)"
    install_path_param="${input_install_path}"
    if [ "$hetero_arch" = "y" ]; then
        if [ "$pkg_is_multi_version" = "true" ]; then
            install_path_param="$install_path_param/$pkg_version_dir/$arch_scripts_path"
        else
            install_path_param="$install_path_param/$arch_scripts_path"
        fi
    fi
fi

if [ "$is_docker_install" = "y" ]; then
    pkg_install_path=$(concat_docker_install_path "${docker_root}" "${install_path_param}")
else
    pkg_install_path="${install_path_param}"
fi

if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" != "y" ]; then
    default_dir="${pkg_install_path}/$pkg_version_dir/compiler"
else
    default_dir="${pkg_install_path}/compiler"
fi
install_info="${default_dir}/ascend_install.info"

if [ "$uninstall" = "y" ]; then
    start_uninstall_log
else
    start_install_log
fi

if [ "$hetero_arch" = "y" ]; then
    log "INFO" "package is running in hetero arch mode!"
fi

# 执行预检查
if [ "$input_pre_check" = "y" ]; then
    log "INFO" "Compiler do pre check started."
    pre_check
    if [ $? -ne 0 ]; then
        log "WARNING" "Compiler do pre check failed."
    else
        log "INFO" "Compiler do pre check finished."
    fi
    if [ "$full_install" = "n" ] && [ "$run_install" = "n" ] && [ "$devel_install" = "n" ] && [ "$upgrade" = "n" ]; then
        exit_install_log 0
    fi
fi

# 版本兼容性检查
if [ "$check" = "y" ]; then
    ver_check
    if [ -z "$pkg_version_dir" ]; then
        preinstall_check --install-path="$install_path_param" --script-dir="$curpath" --package="compiler" --logfile="$logfile" --docker-root="$docker_root"
        if [ $? -ne 0 ]; then
            exit_install_log 1
        else
            log "INFO" "version compatibility check successfully!"
        fi
    fi
    if [ "$full_install" = "n" ] && [ "$run_install" = "n" ] && [ "$devel_install" = "n" ] && [ "$upgrade" = "n" ]; then
        exit_install_log 0
    fi
elif [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ] || [ "$upgrade" = "y" ]; then
    ver_check
    if [ -z "$pkg_version_dir" ]; then
        preinstall_process --install-path="$install_path_param" --script-dir="$curpath" --package="compiler" --logfile="$logfile" --docker-root="$docker_root"
        if [ $? -ne 0 ]; then
            exit_install_log 1
        else
            log "INFO" "version compatibility check successfully!"
        fi
    fi
fi

##################################################################
# 安装升级运行态时, 1/2包必须已安装, 且指定的用户必须存在且与1/2包同属组
if [ "$input_install_for_all" = "n" ]; then
    if [ "$run_install" = "y" ] || [ "$full_install" = "y" ] || [ "$devel_install" = "y" ]; then
        confirm=n
        if [ ! -f "$install_info_old" ]; then
            log "WARNING" "driver and firmware is not exists, please install first."
            confirm=y
        elif [ $(grep -c -i "Driver" "${install_info_old}") -eq 0 ]; then
            log "WARNING" "driver is not exists, please install first."
            confirm=y
        elif [ $(grep -c -i "Firmware" "${install_info_old}") -eq 0 ]; then
            log "WARNING" "firmware is not exists, please install first. (docker scenes is not need)"
            confirm=y
        elif [ -f "${install_info_old}" ]; then
            usergroup_base=$(grep -i usergroup= "${install_info_old}" | cut -d"=" -f2-)
            check_group "${usergroup_base}" "${username}"
            if [ $? -ne 0 ]; then
                log "ERROR" "ERR_NO:0x0093;ERR_DES:User is not belong to the dirver or firmware's installed usergroup! Please add the user (${username}) to the group (${usergroup_base})."
                confirm=y
                exit_install_log 1
            fi
        fi
    fi
fi

uninstall_none_multi_version "$pkg_install_path/compiler"
check_install_for_all
create_default_install_dir_for_common_user
log_base_version
if [ "$upgrade" != "y" ] || [ "$hetero_arch_pkg_installed" != "installed-hetero-to-be-upgraded" ]; then
    is_valid_path
fi
[ "$hetero_arch" = "y" ] && replace_filelist

if [ "$full_install" = "y" ] || [ "$run_install" = "y" ] || [ "$devel_install" = "y" ]; then
    create_default_dir
fi

# 环境上是否已安装过本包
version_installed="$(get_version_installed)"
if [ "x$version_installed" != "x" -a "$version_installed" != "none" ] || [ -f "${install_info}" ]; then
    # 卸载场景
    if [ "$uninstall" = "y" ]; then
        unchattr_files
        uninstall_run "uninstall" "y" "y"
        if [ -d "${default_dir}/site-packages" ]; then
            rm -rf "${default_dir}/site-packages"
        fi
        save_user_files_to_log "$default_dir"
        save_user_files_to_log "$(dirname $default_dir)/atc"
        save_user_files_to_log "$(dirname $default_dir)/fwkacllib"
        exit_uninstall_log 0
    # 升级场景
    elif [ "$upgrade" = "y" ]; then
        if [ -n "$pkg_version_dir" ]; then
            if [ "$hetero_arch" = "y" ]; then
                get_package_upgrade_install_info_hetero "upgrade_install_info"
            else
                get_package_upgrade_install_info "upgrade_install_info" "$pkg_install_path" "compiler"
            fi
            if [ -z "$upgrade_install_info" ]; then
                log "ERROR" "Can not find softlink for this package in latest directory, upgrade failed"
                log_operation "Upgrade" "failed"
                exit_install_log 1
            elif [ "$(realpath $upgrade_install_info)" != "$(realpath $install_info)" ]; then
                uninstall_run "uninstall" "n" "y" "$upgrade_install_info"
            fi
        fi
        unchattr_files
        uninstall_run "uninstall" "n" "n"
        save_user_files_to_log "$default_dir"
        save_user_files_to_log "$(dirname $default_dir)/atc"
        save_user_files_to_log "$(dirname $default_dir)/fwkacllib"
        upgrade_run "upgrade"
        exit_install_log 0
    # 安装场景
    elif [ "$run_install" = "y" ] || [ "$full_install" = "y" ] || [ "$devel_install" = "y" ]; then
        version_of_package=$(get_version_in_runpkg)
        if [ "$is_quiet" = "n" ]; then
            log "INFO" "Compiler package has been installed on the path ${pkg_install_path}, the version is ${version_installed}, and the version of this package is ${version_of_package}, do you want to continue? [y/n]"
            while true
            do
                read yn
                if test "$yn" = n; then
                    log "INFO" "stop installation!"
                    exit_install_log 0
                elif test "$yn" = y; then
                    break
                else
                    log "ERROR" "ERR_NO:0x0002;ERR_DES:input error, please input again!"
                fi
            done
        fi
        unchattr_files
        uninstall_run "uninstall" "n" "n"
        save_user_files_to_log "$default_dir"
        save_user_files_to_log "$(dirname $default_dir)/atc"
        save_user_files_to_log "$(dirname $default_dir)/fwkacllib"
        install_run "install"
        exit_install_log 0
    fi
else
    # 卸载场景
    if [ "$uninstall" = "y" ]; then
        if [ -d "$default_dir" ]; then
            log "ERROR" "The current user does not have the required permission to uninstall $default_dir, uninstall failed"
            log_operation "Uninstall" "failed"
            exit_uninstall_log 1
        else
            log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed in ${pkg_install_path}, uninstall failed"
            log_operation "Uninstall" "failed"
            exit_uninstall_log 1
        fi
    # 升级场景
    elif [ "$upgrade" = "y" ]; then
        if [ -z "$pkg_version_dir" ]; then
            if [ -d "$default_dir" ]; then
                log "ERROR" "The current user does not have the required permission to uninstall $default_dir, upgrade failed"
                log_operation "Upgrade" "failed"
                exit_install_log 1
            else
                log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed in ${pkg_install_path}, upgrade failed"
                log_operation "Upgrade" "failed"
                exit_install_log 1
            fi
        else
            if [ "$hetero_arch" = "y" ]; then
                get_package_upgrade_install_info_hetero "upgrade_install_info"
            else
                get_package_upgrade_install_info "upgrade_install_info" "$pkg_install_path" "compiler"
            fi
            if [ -f "$upgrade_install_info" ]; then
                create_default_dir && cp "$upgrade_install_info" "$install_info"
                migrate_user_assets_v2
                [ "$hetero_arch" = "y" ] && update_install_info_hetero "$install_info" "$pkg_version_dir"
                uninstall_run "uninstall" "n" "y" "$upgrade_install_info"
                upgrade_run "upgrade"
                exit_install_log 0
            else
                log "ERROR" "ERR_NO:0x0080;ERR_DES:Runfile is not installed in ${pkg_install_path}, upgrade failed"
                log_operation "Upgrade" "failed"
                exit_install_log 1
            fi
        fi
    # 安装场景
    elif [ "$run_install" = "y" ] || [ "$full_install" = "y" ] || [ "$devel_install" = "y" ]; then
        install_run "install"
        exit_install_log 0
    fi
fi
