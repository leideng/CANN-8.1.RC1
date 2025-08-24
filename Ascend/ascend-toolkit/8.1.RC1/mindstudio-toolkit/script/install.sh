#!/bin/bash
USERNAME=$(id -un)
USERGROUP=$(id -gn)

readonly PATH_LENGTH=4096

SHELL_DIR=$(cd "$(dirname "$0")" || exit; pwd)
COMMON_PARSER_PATH=${SHELL_DIR}"/parser_install.sh"
COMMON_SHELL_PATH=${SHELL_DIR}"/common.sh"
FILELIST_CSV_PATH=${SHELL_DIR}"/filelist.csv"

function check_params() {
    if [ ! x"${input_install_path}" = "x" ]; then
        check_install_path_valid "${input_install_path}"
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "The install path is invalid, " \
                "only characters in [a-z,A-Z,0-9,-,_] are supported!"
            exit_log 1
        fi
    fi

    # 输入路径长度限制
    if [ ${#input_install_path} -gt ${PATH_LENGTH} ]; then
        log_and_print $LEVEL_ERROR "The install path length is too long!"
        exit_log 1
    fi

    if [ -L "${input_install_path}" ]; then
        log_and_print $LEVEL_ERROR "The install path is softlink, not support"
        exit_log 1
    fi
}

function check_operation() {
    if [ ${version_flag} = y ]; then
        log_and_print $LEVEL_INFO $(cat $SHELL_DIR/../version.info)
        # only query version
        if [ ${install_flag} = n -a ${upgrade_flag} = n -a ${uninstall_flag} = n ]; then
            exit_log 0
        fi
    fi

    if [ ${check_flag} = y ]; then
        if [ ${install_flag} = n -a ${upgrade_flag} = n -a ${uninstall_flag} = n ]; then
            exit_log 0
        fi
    fi

    if [ ! ${install_flag} = y ] && [ ! ${upgrade_flag} = y ] && [ ! ${uninstall_flag} = y ]; then
        log_and_print $LEVEL_ERROR "ERR_NO:0x0004;ERR_DES: Operation failed, please specify install type!"
        exit_log 1
    fi

    if [ ${install_flag} = y -a ${upgrade_flag} = y ] || [ ${install_flag} = y -a ${uninstall_flag} = y ] \
            || [ ${upgrade_flag} = y -a ${uninstall_flag} = y ]; then
        log_and_print $LEVEL_ERROR "ERR_NO:0x0004;ERR_DES: Unsupported parameters, operation failed."
        exit_log 1
    fi
}

function convert_mode() {
    local _mode=$1
    local _new_mode

    local _mode_len=$(expr length ${_mode})
    if [ ${_mode_len} -eq 3 ]; then
        _new_mode=${_mode}
    fi

    # this is for debian
    if [ ${_mode_len} -eq 4 ]; then
        _new_mode=${_mode:1:3}
    fi
    echo ${_new_mode}
}

function check_parent_dir_permission(){
    local _cur_path=$1
    local _ppath=$(dirname ${_cur_path})
    local _cur_dir=$(basename ${_cur_path})

    if [ ! -d ${_cur_path} ]; then
        check_parent_dir_permission ${_ppath}
        return $?
    fi

    if [ "${_cur_path}"x = "/"x ]; then
        log $LEVEL_INFO "check_parent_dir_permission success"
        return 0
    fi

    local _owner=$(stat -c %U "${_ppath}/${_cur_dir}")
    if [ ${_owner} != "root" ]; then
        log_and_print $LEVEL_WARN "The dir [${_cur_dir}] permision not right, " \
            "it should belong to root."
        return 1
    fi

    local _mode=$(stat -c %a "${_ppath}/${_cur_dir}")
    _mode=$(convert_mode ${_mode})
    if [ ${_mode} -lt 755 ]; then
        log_and_print $LEVEL_WARN "The dir [${_cur_dir}] permission is too small, " \
            "it is recommended that the permission be 755 for the root user."
        return 2
    fi

    if [ ${_mode} -gt 755 ]; then
        log_and_print $LEVEL_WARN "The dir [${_cur_dir}] permission is too high, " \
            "it is recommended that the permission be 755 for the root user."
        [ ${quiet_flag} = n ] && return 3
    fi
    check_parent_dir_permission "${_ppath}"
}

function install_path_should_belong_to_root() {
    local _ret=0

    if [ "$(id -u)" -ne 0 ] || [ $uninstall_flag = y ]; then
        return 0
    fi

    check_parent_dir_permission "${install_path}" && _ret=$? || _ret=$?

    # --quiet
    if [ ${quiet_flag} = y ] && [ ${_ret} -ne 0 ]; then
        log_and_print $LEVEL_ERROR "The given dir, or its parents, permission is invalid."
        return 1
    fi

    if [ ${_ret} -ne 0 ]; then
        print_log $LEVEL_INFO "You are going to install on a unsecure path, " \
            "do you want to continue? [y/n]"
        while true; do
            read yn
            [ ${yn} = n ] && return 1
            [ ${yn} = y ] && break
            print_log $LEVEL_ERROR "Input error, please input again!"
        done
    fi
    return 0
}

function convert_install_path() {
    local _install_path="$1"

    # delete last "/" "/."
    _install_path=`echo "${_install_path}" | sed -r "s/((\/)|(\/\.))*$//g"`
    if [ -z "${_install_path}" ]; then
        _install_path="/"
    fi
    # covert relative path to absolute path
    local _prefix=`echo "${_install_path}" | cut -d"/" -f1`
    if [ ! -z "${_prefix}" ] && [ "~" != "${_prefix}" ]; then
        _install_path="${run_path}/${_install_path}"
    fi
    # covert '~' to home path
    local _suffix_path=`echo "${_install_path}" | cut -d"~" -f2`
    if [ "${_suffix_path}" != "${_install_path}" ]; then
        local _home_path=`eval echo "~" | sed "s/\/*$//g"`
        _install_path="${_home_path}${_suffix_path}"
    fi
    echo "${_install_path}"
}

function install_version_path() {
    local _version
    if [ -s ${SHELL_DIR}/../version_name.info ]; then
        _version=$(grep "^version_name=" "${SHELL_DIR}/../version_name.info" | cut -d"=" -f2)
    else
        _version=$(grep "^Version=" "${SHELL_DIR}/../version.info" | cut -d"=" -f2)
    fi
    echo "${_version}"
}

function default_install_path() {
    local _install_path
    if [ "$(id -u)" -eq 0 ]; then
        _install_path="/usr/local/Ascend"
    else
        local _home_path=`eval echo "~"`
        _home_path=`echo "${_home_path}" | sed "s/\/*$//g"`
        _install_path="${_home_path}/Ascend"
    fi
    echo "${_install_path}"
}

function real_install_path() {
    if [ -z "${input_install_path}" ]; then
        input_install_path=${ASCEND_HOME_PATH}
        if [ -z "${input_install_path}" ]; then
            input_install_path=$(default_install_path)
        fi
    fi
    
    install_path=$(convert_install_path "${input_install_path}")

    install_path_owner=$(stat -c %U "${install_path}")
    current_user=$(whoami)
    if [ ! "$install_path_owner" == "$current_user" ]; then
        log_and_print $LEVEL_ERROR "The path ${install_path} is not belong to ${current_user}"
        exit_log 1
    fi

    latest_path="${install_path}/latest"
    origin_install_path=${install_path}
    install_path="${install_path}/$(install_version_path)"

    check_install_path_valid "${install_path}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "The path $install_path is invalid, " \
            "only characters in [a-z,A-Z,0-9,-,_] are supported!"
        exit_log 1
    fi

    install_path_should_belong_to_root
    if [ $? -ne 0 ]; then
        exit_log 1
    fi
}

function update_install_info() {
    if [ -L ${install_file} ] || [ ! -f ${install_file} ]; then
        rm -rf ${install_file} >/dev/null 2>&1
    fi
    if [ ! -f ${install_file} ]; then
        touch ${install_file}
        [ $? -ne 0 ] && exit_log 1
    fi
    change_file_mode 640 ${install_file}

    update_install_param "UserName" "${USERNAME}" ${install_file}
    update_install_param "UserGroup" "${USERGROUP}" ${install_file}
    update_install_param "Install_Path_Param" "${install_path}" ${install_file}

    change_file_mode 440 ${install_file}
}

function install_tool() {
    "$COMMON_PARSER_PATH" --makedir --package=mindstudio-toolkit --username="${USERNAME}" --usergroup="${USERGROUP}" \
        "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Create mindstudio-toolkit subfolders failed."
        return 1
    fi

    "$COMMON_PARSER_PATH" --copy --package=mindstudio-toolkit --username="${USERNAME}" --usergroup="${USERGROUP}" \
        --install_for_all=${install_for_all} "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Copy mindstudio-toolkit files failed."
        return 1
    fi

    "$COMMON_PARSER_PATH" --chmoddir --package=mindstudio-toolkit --username="${USERNAME}" --usergroup="${USERGROUP}" \
        --install_for_all=${install_for_all} "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Chmod mindstudio-toolkit files failed."
        return 1
    fi

    msptiWhl=${install_path}/tools/mspti/python/mspti-0.0.1-py3-none-any.whl
    installWhlPackage $pylocal $msptiWhl ${install_path}/python/site-packages
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "install mindstudio-toolkit mspti whl failed."
        return 1
    fi

    if [ ! -f ${install_path}"/cann_uninstall.sh" ]; then
        cp ${install_path}"/mindstudio-toolkit/script/cann_uninstall.sh" ${install_path}
    fi
    chmod -Rf 500 "${install_path}/mindstudio-toolkit/script"

    return 0
}

function restoremod_file() {
    if [ -d ${install_path}/python/site-packages/mspti-0.0.1.dist-info/ ]; then
        chmod u+w ${install_path}/python/site-packages/mspti-0.0.1.dist-info/
    fi
}

function uninstall_tool() {
    restoremod_file
    whlUninstallPackage mspti ${install_path}/python/site-packages
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Remove mindstudio-toolkit mspti whl failed in ${install_path}."
        return 1
    fi
    # when normal user uninstall package, shell need to restore dir permission
    "$COMMON_PARSER_PATH" --restoremod --package=mindstudio-toolkit --username="unknown" --usergroup="unknown" \
        "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Restore directory written permission failed."
        return 1
    fi

    "$COMMON_PARSER_PATH" --remove --package=mindstudio-toolkit "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "ERR_NO:0X0090;ERR_DES: Remove mindstudio-toolkit files failed in ${install_path}."
        return 1
    fi
    log $LEVEL_INFO "Remove mindstudio-toolkit files succeed in ${install_path}!"
    return 0
}

function install() {
    local _opt_type="install"
    [ ${upgrade_flag} = y ] && _opt_type="upgrade"

    ascend_path=$(dirname "$install_path")
    if [ ! -d "$ascend_path" ]; then
        create_install_path $(dirname "$install_path")
    else
        check_dir_permission "$ascend_path"
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "The path($ascend_path) has no write permission, please check."
            return 1
        fi
    fi

    create_install_path ${install_path}
    create_install_path ${install_path}"/mindstudio-toolkit"
    update_install_info

    local _custom_latest_shell=${latest_path}/mindstudio-toolkit/script/custom_latest.sh
    if [ -f "${_custom_latest_shell}" ]; then
        local _custom_latest_shell_realpath=$(realpath "${_custom_latest_shell}")
        "$_custom_latest_shell_realpath" --uninstall "${latest_path}"
    else
        uninstall_latest "${latest_path}"
    fi

    install_tool && install_latest ${latest_path} ${install_path} ${install_for_all} && register_uninstall ${install_path}
    if [ $? -eq 0 ]; then
        log_and_print $LEVEL_INFO "InstallPath: $install_path"
        log_and_print $LEVEL_INFO "Mindstudio toolkit package ${_opt_type} success! The new version takes effect immediately."
        echo "Please make sure that"
        echo "        - PATH set with ${install_path}/bin"
        echo "        - LD_LIBRARY_PATH set with ${install_path}/lib64"
        return 0
    else
        log_and_print ${LEVEL_ERROR} "Mindstudio toolkit ${_opt_type} failed, try to fallback..."
        # roll back
        local _is_fallback=y
        uninstall $_is_fallback > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log_and_print ${LEVEL_ERROR} "Fallback failed."
        else
            log_and_print ${LEVEL_INFO} "Fallback succeed."
        fi
        return 1
    fi
}

function uninstall() {
    local _is_fallback=n
    [ ! -z $1 ] && _is_fallback=$1

    should_latest_uninstall ${latest_path}
    local _is_latest_uninstall=$?
    if [ $_is_fallback = y -o ${_is_latest_uninstall} -eq 0 ]; then
        local _custom_latest_shell=${latest_path}/mindstudio-toolkit/script/custom_latest.sh
        if [ -f "${_custom_latest_shell}" ]; then
            local _custom_latest_shell_realpath=$(realpath "${_custom_latest_shell}")
            "$_custom_latest_shell_realpath" --uninstall "${latest_path}"
        else
            uninstall_latest ${latest_path}
        fi
        if [ $? -ne 0 ]; then
            log_and_print ${LEVEL_ERROR} "Mindstudio toolkit uninstall lastest failed, please retry."
            return 1
        fi
    fi

    uninstall_tool
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "Mindstudio toolkit uninstall failed, please retry."
        return 1
    fi

    rm -f ${install_file}
    is_dir_empty ${install_path}"/mindstudio-toolkit"
    if [ $? -eq 0 ]; then
        rm -rf ${install_path}"/mindstudio-toolkit"
    fi
    unregister_uninstall ${install_path}
    remove_uninstall_file_if_no_content ${install_path}"/cann_uninstall.sh"
    if [ $_is_fallback = y -o ${_is_latest_uninstall} -eq 0 ]; then
        update_latest_sortlink ${latest_path}
    fi
    remove_empty_dir ${install_path}
    remove_empty_dir $(dirname ${install_path})
    log_and_print $LEVEL_INFO "Mindstudio toolkit uninstall success!"
    return 0
}

function upgrade() {
    if [ ${install_flag} = y ] && [ ! ${quiet_flag} = y ]; then
        print_log $LEVEL_INFO "Mindstudio toolkit package has been installed on the path $install_path, " \
            "do you want to overwrite current installation? [y/n] "
        while true; do
            read yn
            [ ${yn} = n ] && return 0
            [ ${yn} = y ] && break
            print_log $LEVEL_ERROR "Input error, please input again!"
        done
    fi

    uninstall
    [ $? -ne 0 ] && return 1

    install
    [ $? -ne 0 ] && return 1
    return 0
}

function get_uninstall_path() {
    local _latest_link_path=$(readlink -f "${latest_path}/mindstudio-toolkit")
    echo "${_latest_link_path}/script"
}

function get_uninstall_msprof_path() {
    if [ -f "$install_file" ];then
        echo "${install_path}/mindstudio-msprof/script"
    else
        local _latest_link_path=$(readlink -f "${latest_path}/mindstudio-msprof")
        echo "${_latest_link_path}/script"
    fi
}

function uninstall_msprof() {
    local msprof_script_path="$(get_uninstall_msprof_path)/uninstall.sh"
    if [ -f "${msprof_script_path}" ];then
        bash "${msprof_script_path}"
        if [ $? -ne 0 ];then
            log_and_print $LEVEL_ERROR "Uninstall mindstudio msprof failed."
            return 1
        fi
    fi
    return 0
}

function install_pkg() {
    install_file=${install_path}"/mindstudio-toolkit/ascend_install.info"
    if [ $(id -u) -eq 0 ]; then
            install_for_all=y
    fi

    if [ -f ${install_file} ]; then
        if [ ${uninstall_flag} = y ]; then
            uninstall
        elif [ ${install_flag} = y ]; then
            uninstall_msprof
            upgrade
        elif [ ${upgrade_flag} = y ]; then
            uninstall_msprof
            bash "$(get_uninstall_path)/uninstall.sh"
            [ $? -ne 0 ] && exit_log 1
            install
        fi
    else
        if [ ${uninstall_flag} = y ]; then
            log_and_print $LEVEL_ERROR "Package is not installed " \
                "on the path $install_path, uninstall failed."
        elif [ ${install_flag} = y ]; then
            install
        elif [ ${upgrade_flag} = y ]; then
            local _link_path=$(dirname $(readlink -f ${latest_path}/mindstudio-toolkit))
            uninstall_msprof
            bash "$(get_uninstall_path)/uninstall.sh"
            [ $? -ne 0 ] && exit_log 1
            remove_empty_dir "${_link_path}/${ARCH}-${OS}/include"
            install
        fi
    fi
}

function execute_with_lock() {
    local lock_path="$1"
    local lockfile="${lock_path}/.mindstudio.lock"
    touch "$lockfile"
    local lock_command="$2"
    (
        flock -n 9
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "There is already package being executed, " \
                        "please do not execute repeatedly at the same time!"
            exit_log 1
        fi
        eval "$lock_command"
    ) 9>"$lockfile"

    rm -f "$lockfile"
}

function main() {
    check_operation
    real_install_path
    execute_with_lock "${origin_install_path}" install_pkg
}

source ${COMMON_SHELL_PATH}

install_file=""
input_install_path=""
install_for_all=n
install_flag=n
uninstall_flag=n
upgrade_flag=n
quiet_flag=n
version_flag=n
check_flag=n
pylocal=y

# get run package path
run_path=`echo "$2" | cut -d"-" -f3-`
if [ -z "${run_path}" ]; then
    run_path=`pwd`
else
    # delete last "/" "/."
    run_path=`echo "${run_path}" | sed "s/((\/)|(\/\.))*$//g"`
    [ -z "${run_path}" ] && run_path="/"
    if [ ! -d "${run_path}" ]; then
        log_and_print $LEVEL_ERROR "Run package path is invalid: $run_path"
        exit 1
    fi
fi

shift 2

start_log

for arg in "$@"; do
    case "$arg" in
    --check)
        check_flag=y
        ;;
    --quiet)
        quiet_flag=y
        ;;
    --install-path=*)
        input_install_path=`echo "$arg" | cut -d"=" -f2`
        ;;
    --install-for-all)
        install_for_all=y
        ;;
    --full)
        install_flag=y
        ;;
    --run)
        install_flag=y
        ;;
    --devel)
        install_flag=y
        ;;
    --uninstall)
        uninstall_flag=y
        ;;
    --upgrade)
        upgrade_flag=y
        ;;
    --version)
        version_flag=y
        ;;
    -*)
        log_and_print $LEVEL_ERROR "Unsupported parameters: $arg"
        exit_log 1
        ;;
    *)
        if [ ! -z "$arg" ]; then
            log_and_print $LEVEL_ERROR "Unsupported parameters: $arg"
            exit_log 1
        fi
        break
        ;;
    esac
done

check_params

# print param
log_and_print $LEVEL_INFO "LogFile: $log_file"
log_and_print $LEVEL_INFO "InputParams: $*"

main

exit_log 0
