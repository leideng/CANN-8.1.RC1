#!/bin/bash
INSTALL_INFO_KEY_ARRAY=("UserName" "UserGroup" "Install_Path_Param")
MODULE_NAME="Mindstudio-toolkit"
LEVEL_ERROR="ERROR"
LEVEL_WARN="WARNING"
LEVEL_INFO="INFO"
USERNAME=$(id -un)
USERGROUP=$(id -gn)
SHELL_DIR=$(cd "$(dirname "$0")" || exit; pwd)

ARCH=$(cat $SHELL_DIR/../scene.info | grep arch | cut -d"=" -f2)
OS=$(cat $SHELL_DIR/../scene.info | grep os | cut -d"=" -f2)

MINDSTUDIO_ARRAY=("mindstudio-toolkit")
LINK_ARRAY=("bin" "lib64")

export log_file=""

function log() {
    local content=`echo "$@" | cut -d" " -f2-`
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content" >> "${log_file}"
}

function log_and_print() {
    local content=`echo "$@" | cut -d" " -f2-`
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content"
    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content" >> "${log_file}"
}

function print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content"
}

function init_log() {
    local _log_path="/var/log/ascend_seclog"
    local _log_file="ascend_install.log"

    if [ $(id -u) -ne 0 ]; then
        local _home_path=`eval echo "~"`
        _log_path="${_home_path}${_log_path}"
    fi

    log_file="${_log_path}/${_log_file}"

    create_folder "${_log_path}" "${USERNAME}:${USERGROUP}" 750
    if [ $? -ne 0 ]; then
        print_log $LEVEL_WARN "Create ${_log_path} failed."
    fi

    if [ -L "${log_file}" ] || [ ! -f "${log_file}" ]; then
        rm -rf "${log_file}" >/dev/null 2>&1
    fi
    create_file "${log_file}" "${USERNAME}:${USERGROUP}" 640
    if [ $? -ne 0 ]; then
        print_log $LEVEL_WARN "Create ${log_file} failed."
    fi
}

function start_log() {
    free_log_space
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: Start Time: $cur_date"
    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: Start Time: $cur_date" >> "${log_file}"
}

function exit_log() {
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: End Time: $cur_date"
    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: End Time: $cur_date" >> "${log_file}"
    exit $1
}

function free_log_space() {
    local file_size=$(stat -c %s "${log_file}")
    # mindstudio install log file will be limited in 20M
    if [ "${file_size}" -gt $((1024 * 1024 * 20)) ]; then
        local ibs=512
        local delete_size=$((${file_size} / 3 / ${ibs}))
        dd if="$log_file" of="${log_file}_tmp" bs="${ibs}" skip="${delete_size}" > /dev/null 2>&1
        mv "${log_file}_tmp" "${log_file}"
        chmod 640 ${log_file}
    fi
}

function update_install_param() {
    local _key=$1
    local _val=$2
    local _file=$3
    local _param

    if [ ! -f "${_file}" ]; then
        exit 1
    fi

    for key_param in "${INSTALL_INFO_KEY_ARRAY[@]}"; do
        if [ ${key_param} != ${_key} ]; then
            continue
        fi
        _param=`grep -r "${_key}=" "${_file}"`
        if [ "x${_param}" = "x" ]; then
            echo "${_key}=${_val}" >> "${_file}"
        else
            sed -i "/^${_key}=/c ${_key}=${_val}" "${_file}"
        fi
        break
    done
}

function get_install_param() {
    local _key=$1
    local _file=$2
    local _param

    if [ ! -f "${_file}" ];then
        exit 1
    fi

    for key_param in "${INSTALL_INFO_KEY_ARRAY[@]}"; do
        if [ ${key_param} != ${_key} ]; then
            continue
        fi
        _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
        break
    done
    echo "${_param}"
}

function change_mode() {
    local _mode=$1
    local _path=$2
    local _type=$3

    if [ ! x"${install_for_all}" = "x" ] && [ ${install_for_all} = y ]; then
        _mode="$(expr substr ${_mode} 1 2)$(expr substr ${_mode} 2 1)"
    fi
    if [ ${_type} = "dir" ]; then
        find "${_path}" -type d -exec chmod ${_mode} {} \; 2> /dev/null
    elif [ ${_type} = "file" ]; then
        find "${_path}" -type f -exec chmod ${_mode} {} \; 2> /dev/null
    fi
}

function change_file_mode() {
    local _mode=$1
    local _path=$2
    change_mode ${_mode} "${_path}" file
}

function change_dir_mode() {
    local _mode=$1
    local _path=$2
    change_mode ${_mode} "${_path}" dir
}

function create_file() {
    local _file=$1

    if [ ! -f "${_file}" ]; then
        touch "${_file}"
        [ $? -ne 0 ] && return 1
    fi

    chown -hf "$2" "${_file}"
    [ $? -ne 0 ] && return 1
    change_file_mode "$3" "${_file}"
    [ $? -ne 0 ] && return 1
    return 0
}

function create_folder() {
    local _path=$1

    if [ -z ${_path} ]; then
        return 1
    fi

    if [ ! -d ${_path} ]; then
        mkdir -p ${_path} >/dev/null 2>&1
        [ $? -ne 0 ] && return 1
    fi

    chown -hf $2 ${_path}
    [ $? -ne 0 ] && return 1
    change_dir_mode $3 ${_path}
    [ $? -ne 0 ] && return 1
    return 0
}

function is_dir_empty() {
    local _path=$1
    local _file_num

    if [ -z ${_path} ]; then
        return 1
    fi

    if [ ! -d ${_path} ]; then
        return 1
    fi
    _file_num=`ls "${_path}" | wc -l`
    if [ ${_file_num} -eq 0 ]; then
        return 0
    fi
    return 1
}

function check_install_path_valid() {
    local install_path="$1"
    # 黑名单设置，不允许//，...这样的路径
    if echo "${install_path}" | grep -Eq '\/{2,}|\.{3,}'; then
        return 1
    fi
    # 白名单设置，只允许常见字符
    if echo "${install_path}" | grep -Eq '^\~?[a-zA-Z0-9./_-]*$'; then
        return 0
    else
        return 1
    fi
}

function check_dir_permission() {
    local _path=$1

    if [ -z ${_path} ]; then
        log_and_print $LEVEL_ERROR "The dir path is empty."
        exit 1
    fi

    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi

    if [ -d ${_path} ] && [ ! -w ${_path} ]; then
        return 1
    fi

    return 0
}

function create_relative_softlink() {
    local _src_path="$1"
    local _des_path="$2"

    local _des_dir_name=$(dirname $_des_path)
    _src_path=$(readlink -f ${_src_path})
    if [ ! -f "$_src_path" -a ! -d "$_src_path" -a ! -L "$_src_path" ]; then
        return
    fi
    _src_path=$(get_relative_path $_des_dir_name $_src_path)
    if [ -L "${_des_path}" ]; then
        delete_softlink "${_des_path}"
    fi
    ln -sf "${_src_path}" "${_des_path}"
    if [ $? -ne 0 ]; then
        print_log $LEVEL_ERROR "${_src_path} softlink to ${_des_path} failed!"
        exit 1
    fi
}

function delete_softlink() {
    local _path="$1"
    # 如果目标路径是个软链接，则移除
    if [ -L "${_path}" ]; then
        local _parent_path=$(dirname ${_path})
        if [ ! -w ${_parent_path} ]; then
            chmod u+w ${_parent_path}
            rm -f "${_path}"
            if [ $? -ne 0 ]; then
                print_log $LEVEL_ERROR "remove softlink ${_path} failed!"
                exit 1
            fi
            chmod u-w ${_parent_path}
        else
            rm -f "${_path}"
            if [ $? -ne 0 ]; then
                print_log $LEVEL_ERROR "remove softlink ${_path} failed!"
                exit 1
            fi
        fi
    fi
}

function create_install_path() {
    local _install_path=$1

    if [ ! -d "${_install_path}" ]; then
        local _ppath=$(dirname ${_install_path})
        while [[ ! -d ${_ppath} ]];do
            _ppath=$(dirname ${_ppath})
        done

        check_dir_permission "${_ppath}"
        if [ $? -ne 0 ]; then
            chmod u+w -R ${_ppath}
            [ $? -ne 0 ] && exit_log 1
        fi

        create_folder "${_install_path}" $USERNAME:$USERGROUP 750
        [ $? -ne 0 ] && exit_log 1
    else
        check_dir_permission "${_install_path}"
        if [ $? -ne 0 ]; then
            chmod u+w -R ${_install_path}
        fi
    fi
}

function remove_empty_dir() {
    if [ -d "$1" ] && [ -z "$(ls -A $1)" ] && [[ ! "$1" =~ ^/+$ ]]; then
        if [ ! -w $(dirname $1) ]; then
            chmod u+w $(dirname $1)
            rm -rf "$1"
            if [ $? != 0 ]; then
                print_log $LEVEL_ERROR "delete directory $1 fail"
                exit 1
            fi
            chmod u-w $(dirname $1)
        else
            rm -rf "$1"
            if [ $? != 0 ]; then
                print_log $LEVEL_ERROR "delete directory $1 fail"
                exit 1
            fi
        fi
    fi
}

function get_relative_path() {
    local _relative_to_path=$1
    local _des_path=$2
    echo $(realpath --relative-to=$_relative_to_path $_des_path)
}

function register_uninstall() {
    local _install_path=$1
    chmod u+w ${_install_path}"/cann_uninstall.sh"
    sed -i "/^exit /i uninstall_package \"mindstudio-toolkit\/script\"" ${_install_path}"/cann_uninstall.sh"
    chmod u-w ${_install_path}"/cann_uninstall.sh"
}

function unregister_uninstall() {
    local _install_path=$1
    if [ -f ${_install_path}"/cann_uninstall.sh" ]; then
        chmod u+w ${_install_path}"/cann_uninstall.sh"
        remove_uninstall_package ${_install_path}"/cann_uninstall.sh"
        chmod u-w ${_install_path}"/cann_uninstall.sh"
    fi
}

# 删除uninstall.sh文件，如果已经没有uninstall_package调用
function remove_uninstall_file_if_no_content() {
    local _file="$1"
    local _num

    if [ ! -f "${_file}" ]; then
        return 0
    fi

    _num=$(grep "^uninstall_package " ${_file} | wc -l)
    if [ ${_num} -eq 0 ]; then
        rm -f "${_file}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_WARN "Delete file:${_file} failed, please delete it by yourself."
        fi
    fi
}

# 删除uninstall.sh文件中的uninstall_package函数调用
function remove_uninstall_package() {
    local _file="$1"

    if [ -f "${_file}" ]; then
        sed -i "/uninstall_package \"mindstudio-toolkit\/script\"/d" "${_file}"
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "remove ${_file} uninstall_package command failed!"
            exit 1
        fi
    fi
}

#根据依赖关系，查找mindstudio-toolkit的指向路径
function update_latest_sortlink() {
    local _latest_path=$1
    if [ -L "${_latest_path}/mindstudio-toolkit" ]; then
        local _mindstudio_latest_path=$(readlink -f "${_latest_path}/mindstudio-toolkit")
        if [ -d ${_mindstudio_latest_path} ]; then
            local _mindstudio_link_path=$(dirname ${_mindstudio_latest_path})
            if [ -d ${_mindstudio_link_path}"/mindstudio-toolkit" ]; then
                log_and_print ${LEVEL_INFO} "Start update latest sortlink."
                local _custom_latest_shell=${_mindstudio_link_path}/mindstudio-toolkit/script/custom_latest.sh
                if [ -f "${_custom_latest_shell}" ]; then
                    local _custom_latest_shell_realpath=$(realpath "${_custom_latest_shell}")
                    "$_custom_latest_shell_realpath" --install "${_latest_path}" "${_mindstudio_link_path}" "n"
                else
                    install_latest "${_latest_path}" "${_mindstudio_link_path}" "n"
                fi
            fi
        fi
    fi
}

function switch_to_the_previous_version() {
    local _latest_path=$1
    if [ ! -L "${_latest_path}/mindstudio-toolkit" ]; then
      if [ -L "${_latest_path}/runtime" ]; then
          local _mindstudio_link_path=$(dirname $(readlink -f ${_latest_path}/runtime))
          create_relative_softlink "${_mindstudio_link_path}/mindstudio-toolkit" "${_latest_path}/mindstudio-toolkit"
      fi
    fi
}

function should_latest_uninstall() {
    local _latest_path=$1
    local _curr_version=$(cat $SHELL_DIR/../version_name.info)
    if [ ! -f ${_latest_path}"/mindstudio-toolkit/version_name.info" ]; then
        return 1
    fi
    local _latest_version=$(cat ${_latest_path}"/mindstudio-toolkit/version_name.info")
    if [ "${_curr_version}" = "${_latest_version}" ]; then
        return 0
    fi
    return 1
}

function change_latest_mod() {
    local _latest_path=$1
    local _install_for_all=$2
    if [ "${_install_for_all}" = "y" ]; then
        chmod 555 "${_latest_path}/${ARCH}-${OS}/bin"
        chmod 555 "${_latest_path}/${ARCH}-${OS}/lib64"
        chmod 555 "${_latest_path}/${ARCH}-${OS}"
    else
        chmod 550 "${_latest_path}/${ARCH}-${OS}/bin"
        chmod 550 "${_latest_path}/${ARCH}-${OS}/lib64"
        chmod 550 "${_latest_path}/${ARCH}-${OS}"
    fi
}

function create_python_latest_softlink()
{
    local _latest_path=$1
    local _link_path=$2
    create_relative_softlink "${_link_path}/python/site-packages/msmodelslim" "${_latest_path}/python/site-packages/msmodelslim"
    create_relative_softlink "${_link_path}/python/site-packages/mspti" "${_latest_path}/python/site-packages/mspti"
}

function delete_python_latest_softlink()
{
    local _latest_path=$1
    delete_softlink "${_latest_path}/python/site-packages/msmodelslim"
    delete_softlink "${_latest_path}/python/site-packages/mspti"
}

function create_msprof_latest_softlink()
{
  local _latest_path=$1
  local _link_path=$2
  create_relative_softlink "${_link_path}/tools/profiler/bin/msprof" "${_latest_path}/bin/msprof"
}

function delete_msprof_latest_softlink()
{
  local _latest_path=$1
  delete_softlink "${_latest_path}/bin/msprof"
}

function create_mstx_latest_softlink()
{
  local _latest_path=$1
  local _link_path=$2
  create_relative_softlink "${_link_path}/${ARCH}-${OS}/include/mstx" "${_latest_path}/${ARCH}-${OS}/include/mstx"
}

function delete_mstx_latest_softlink()
{
  local _latest_path=$1
  delete_softlink "${_latest_path}/${ARCH}-${OS}/include/mstx"
}

function create_mspti_latest_softlink()
{
  local _latest_path=$1
  local _link_path=$2
  create_relative_softlink "${_link_path}/${ARCH}-${OS}/include/mspti" "${_latest_path}/${ARCH}-${OS}/include/mspti"
}

function delete_mspti_latest_softlink()
{
  local _latest_path=$1
  delete_softlink "${_latest_path}/${ARCH}-${OS}/include/mspti"
}

function create_msserviceprofiler_latest_softlink()
{
  local _latest_path=$1
  local _link_path=$2
  create_relative_softlink "${_link_path}/${ARCH}-${OS}/include/msServiceProfiler" "${_latest_path}/${ARCH}-${OS}/include/msServiceProfiler"
}

function delete_msserviceprofiler_latest_softlink()
{
  local _latest_path=$1
  delete_softlink "${_latest_path}/${ARCH}-${OS}/include/msServiceProfiler"
}

function install_latest() {
    local _latest_path=$1
    local _link_path=$2
    local _install_for_all=$3

    create_install_path ${_latest_path}
    local _filelist_csv=${SHELL_DIR}"/filelist.csv"
    if [ ! -f ${_filelist_csv} ]; then
        if [ -f "${_link_path}/mindstudio-toolkit/script/filelist.csv" ]; then
            _filelist_csv="${_link_path}/mindstudio-toolkit/script/filelist.csv"
        else
            log_and_print $LEVEL_WARN "File filelist.csv doesn't exist."
            return 1
        fi
    fi

    create_folder "${_latest_path}/${ARCH}-${OS}/lib64" $USERNAME:$USERGROUP 750
    for element in $(cat ${_filelist_csv} | grep "${ARCH}-${OS}/lib64" \
            | grep ".so" | awk -F "\"*,\"*" '{print $3}'); do
        element=$(basename ${element})
        create_relative_softlink "${_link_path}/${ARCH}-${OS}/lib64/${element}" \
            "${_latest_path}/${ARCH}-${OS}/lib64/${element}"
    done

    create_folder "${_latest_path}/${ARCH}-${OS}/bin" $USERNAME:$USERGROUP 750
    for element in $(cat ${_filelist_csv} | grep "${ARCH}-${OS}/bin" \
            | grep "mindstudio" | awk -F "\"*,\"*" '{print $3}'); do
        element=$(basename ${element})
        create_relative_softlink "${_link_path}/${ARCH}-${OS}/bin/${element}" \
            "${_latest_path}/${ARCH}-${OS}/bin/${element}"
    done

    create_folder "${_latest_path}/python/site-packages" $USERNAME:$USERGROUP 750
    for element in $(cat ${_filelist_csv} | grep "python/site-packages" | awk -F "\"*,\"*" '{print $3}' | grep -v NA); do
        element=$(basename ${element})
        create_relative_softlink "${_link_path}/python/site-packages/${element}" \
            "${_latest_path}/python/site-packages/${element}"
    done
    create_python_latest_softlink ${_latest_path} ${_link_path}
    create_msprof_latest_softlink ${_latest_path} ${_link_path}

    create_folder "${_latest_path}/${ARCH}-${OS}/include" $USERNAME:$USERGROUP 750
    create_mstx_latest_softlink ${_latest_path} ${_link_path}

    create_folder "${_latest_path}/${ARCH}-${OS}/include" $USERNAME:$USERGROUP 750
    create_mspti_latest_softlink ${_latest_path} ${_link_path}

    create_folder "${_latest_path}/${ARCH}-${OS}/include" $USERNAME:$USERGROUP 750
    create_msserviceprofiler_latest_softlink ${_latest_path} ${_link_path}
    for element in $(cat ${_filelist_csv} | grep "${ARCH}-${OS}/include" \
            | grep ".so" | awk -F "\"*,\"*" '{print $3}'); do
        element=$(basename ${element})
        create_relative_softlink "${_link_path}/${ARCH}-${OS}/include/${element}" \
            "${_latest_path}/${ARCH}-${OS}/include/${element}"
    done

    create_folder "${_latest_path}/tools" $USERNAME:$USERGROUP 750
    for element in $(cat ${_filelist_csv} | grep tools/ | grep mkdir \
            | awk -F "\"*,\"*" '{print $4}' | awk -F "\"*/\"*" '{print $2}' | uniq); do
        create_relative_softlink "${_link_path}/tools/${element}" "${_latest_path}/tools/${element}"
    done

    for element in ${MINDSTUDIO_ARRAY[@]}; do
        create_relative_softlink "${_link_path}/${element}" "${_latest_path}/${element}"
    done

    for element in ${LINK_ARRAY[@]}; do
        if [ ! -L "${_latest_path}/${element}" ]; then
            create_relative_softlink "${_latest_path}/${ARCH}-${OS}/${element}" "${_latest_path}/${element}"
        fi
    done
    change_latest_mod ${_latest_path} ${_install_for_all}
}

function uninstall_latest() {
    local _latest_path=$1
    if [ ! -L "${_latest_path}/mindstudio-toolkit" ]; then
        return 0
    fi
    local _filelist_csv="${_latest_path}/mindstudio-toolkit/script/filelist.csv"
    if [ ! -f ${_filelist_csv} ]; then
        log_and_print $LEVEL_INFO "File filelist.csv doesn't exist."
        return 0
    fi
    for element in $(cat ${_filelist_csv} | grep "python/site-packages" | awk -F "\"*,\"*" '{print $3}' | grep -v NA); do
        element=$(basename ${element})
        delete_softlink "${_latest_path}/python/site-packages/${element}"
    done
    delete_python_latest_softlink ${_latest_path}
    delete_msprof_latest_softlink ${_latest_path}
    delete_mstx_latest_softlink ${_latest_path}
    delete_mspti_latest_softlink ${_latest_path}
    delete_msserviceprofiler_latest_softlink ${_latest_path}

    remove_empty_dir "${_latest_path}/python/site-packages"
    remove_empty_dir "${_latest_path}/python"

    for element in $(cat ${_filelist_csv} | grep tools/ | grep mkdir \
            | awk -F "\"*,\"*" '{print $4}' | awk -F "\"*/\"*" '{print $2}' | uniq); do
        delete_softlink "${_latest_path}/tools/${element}"
    done
    remove_empty_dir "${_latest_path}/tools"

    for element in $(cat ${_filelist_csv} | grep "${ARCH}-${OS}/bin" \
            | grep "mindstudio" | awk -F "\"*,\"*" '{print $3}'); do
        element=$(basename ${element})
        delete_softlink "${_latest_path}/bin/${element}"
    done
    if [ -z "$(ls -A "${_latest_path}/bin")" ]; then
        delete_softlink "${_latest_path}/bin"
    fi

    for element in $(cat ${_filelist_csv} | grep "${ARCH}-${OS}/lib64" \
            | grep ".so" | awk -F "\"*,\"*" '{print $3}'); do
        element=$(basename ${element})
        delete_softlink "${_latest_path}/lib64/${element}"
    done
    if [ -z "$(ls -A "${_latest_path}/lib64")" ]; then
        delete_softlink "${_latest_path}/lib64"
    fi

    for element in ${MINDSTUDIO_ARRAY[@]}; do
        delete_softlink "${_latest_path}/${element}"
    done
    remove_empty_dir "${_latest_path}/${ARCH}-${OS}/bin"
    remove_empty_dir "${_latest_path}/${ARCH}-${OS}/lib64"
    remove_empty_dir "${_latest_path}/${ARCH}-${OS}/include"
    remove_empty_dir "${_latest_path}/${ARCH}-${OS}"

    remove_empty_dir ${_latest_path}
    return 0
}

installWhlPackage() {
    local _pylocal=$1
    local _package=$2
    local _pythonlocalpath=$3

    log ${LEVEL_INFO} "start to begin install ${_package}."
    if [ ! -f "${_package}" ]; then
        log_and_print ${LEVEL_ERROR} "install whl The ${_package} does not exist."
        return 1
    fi
    if [ "-${_pylocal}" = "-y" ]; then
        pip3 install --upgrade --no-deps --force-reinstall "${_package}" -t "${_pythonlocalpath}" > /dev/null 2>&1
    else
        if [ "$(id -u)" -ne 0 ]; then
            pip3 install --upgrade --no-deps --force-reinstall "${_package}" --user > /dev/null 2>&1
        else
            pip3 install --upgrade --no-deps --force-reinstall "${_package}" > /dev/null 2>&1
        fi
    fi
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "Install ${_package} failed."
        return 1
    fi
    log ${LEVEL_INFO} "install ${_package} succeed."
    return 0
}

### uninstall whl
whlUninstallPackage() {
    local module_="$1"
    local python_path_="$2"

    log ${LEVEL_INFO} "start to uninstall ${module_}"
    export PYTHONPATH=${python_path_}
    pip3 show ${module_} > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log ${LEVEL_WARN} "${module_} has not been installed."
        return 0
    fi
    pip3 uninstall -y "${module_}" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "uninstall ${module_} failed."
        return 1
    fi
    log ${LEVEL_INFO} "uninstall ${module_} succeed."
    return 0
}

init_log
