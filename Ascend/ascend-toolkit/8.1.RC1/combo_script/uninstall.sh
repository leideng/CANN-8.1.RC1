#!/bin/bash

# 请在此处定义各种变量
readonly PACKAGE_SHORT_NAME="toolkit"
readonly PACKAGE_VERSION_FORM="aarch64-linux"
readonly PACKAGE_LOG_NAME="Toolkit"
readonly CANN_INSTALL_CONF="ascend_cann_install.info"

# 路径
script_path="$(dirname $(readlink -f $0))"
form_path="$(cd "$(dirname ${script_path})" && pwd)"
version_path=$(dirname ${form_path})
frame=$(arch)
if [ x"${frame}" == x"aarch64" ]; then
    form_path_same="${form_path}/aarch64-linux"
    cann_uninstall_same="${form_path}/cann_uninstall.sh"
    form_path_diff="${form_path}/x86_64-linux"
    cann_uninstall_diff="${form_path}/x86_64-linux/hetero-arch-scripts/cann_uninstall.sh"
elif [ x"${frame}" == x"x86_64" ]; then
    form_path_same="${form_path}/x86_64-linux"
    cann_uninstall_same="${form_path}/cann_uninstall.sh"
    form_path_diff="${form_path}/aarch64-linux"
    cann_uninstall_diff="${form_path}/aarch64-linux/hetero-arch-scripts/cann_uninstall.sh"
elif [ x"${frame}" == x"sw_64" ]; then
    form_path_same="${form_path}/sw_64-linux"
    cann_uninstall_same="${form_path}/cann_uninstall.sh"
fi

if [ "$UID" = "0" ]; then
    CANN_INSTALL_INFO="/etc/Ascend/${CANN_INSTALL_CONF}"
else
    CANN_INSTALL_INFO="${HOME}/Ascend/${CANN_INSTALL_CONF}"
fi

# 将关键信息打印到屏幕上
function print() {
    echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
}

# 安全删除文件
function fn_del_file() {
    local file_path=$1
    # 判断变量是否为空
    if [ -n "${file_path}" ]; then
        # 判断是否是文件
        if [ -f "${file_path}" ] || [ -h "${file_path}" ]; then
            rm -f "${file_path}"
            print "INFO" " delete file ${file_path} successfully."
            return 0
        elif ls ${file_path} 1>/dev/null 2>&1; then
            if [ -d ${file_path} ]; then
                print "WARNING" " delete operation, the ${file_path} is directory, not file."
                return 1
            else
                rm -f "${file_path}"
                print "INFO" " delete wildcard file ${file_path} successfully."
                return 0
            fi
        else
            print "WARNING" " delete operation, the file ${file_path} is not exist."
            return 1
        fi
    else
        print "WARNING" " delete operation, file parameter invalid."
        return 2
    fi
}

# 安全删除文件夹
function fn_del_dir() {
    local dir_path=$1
    local is_empty=$2
    # 判断变量不为空且不是系统根盘
    if [ -n "${dir_path}" ] && [[ ! "${dir_path}" =~ ^/+$ ]]; then
        # 判断是否是目录
        if [ -d "${dir_path}" ]; then
            # 判断是否需要判断目录为空不删除
            if [ x"${is_empty}" == x ] || [ "$(ls -A ${dir_path})" = "" ]; then
                chmod -R 700 "${dir_path}" 2>/dev/null
                rm -rf "${dir_path}"
                print "INFO" " delete directory ${dir_path} successfully."
                return 0
            else
                print "INFO" " delete operation, the directory ${dir_path} is not empty."
                return 1
            fi
        else
            print "WARNING" " delete operation, the ${dir_path} is not exist or not directory."
            return 1
        fi
    else
        print "WARNING" " delete operation, directory parameter invalid."
        return 2
    fi
}

# 判断spc_uninstall 脚本是否存在并执行
function remove_spc_dir() {
    # 判断卸载脚本是否存在
    local spc_uninstall_script="${form_path}/spc/script/uninstall.sh"
    if [ ! -f ${spc_uninstall_script} ]; then
        return
    fi
    # 执行卸载总脚本
    print "INFO" "uninstall patch start"
    print "INFO" "${spc_uninstall_script}"
    bash ${spc_uninstall_script}
    if [[ $? -ne 0 ]]; then
        print "ERROR" "spc uninstall failed"
    else
        print "INFO" "spc uninstall success"
    fi
}

# 判断并删除配置文件
function del_config_file() {
    local num=0
    local record_change=n
    if [ x"${frame}" == x"sw_64" ]; then
        num=$(grep "^uninstall_package " "${cann_uninstall_same}" | wc -l)
        grep "^uninstall_package \"combo_script" "${cann_uninstall_same}" 1>/dev/null 2>&1
        if [ $? -eq 0 ] && [ ${num} -eq 1 ]; then
            if [ -d ${form_path_same} ]; then
                record_change=n
                if [ ! -w ${form_path_same} ]; then
                    chmod u+w ${form_path_same}
                    record_change=y
                fi
                fn_del_config "${form_path_same}"
                fn_del_file "${form_path_same}/runtime"
                fn_del_dir "${form_path_same}" "true"
                if [ -d ${form_path_same} ] && [ ${record_change} = y ]; then
                    chmod u-w ${form_path_same}
                fi
            fi
            fn_del_file "${cann_uninstall_same}"
            fn_del_dir "${form_path}/combo_script"
            return
        else
            print "ERROR" "The script combo_script/uninstall.sh cannot be operated."
            exit 1
        fi
    fi
    if [ x"${PACKAGE_VERSION_FORM}" == x ]; then
        if [ -f ${cann_uninstall_same} ]; then
            num=$(grep "^uninstall_package " "${cann_uninstall_same}" | wc -l)
            grep "^uninstall_package \"combo_script" "${cann_uninstall_same}" 1>/dev/null 2>&1
            if [ $? -eq 0 ] && [ ${num} -eq 1 ]; then
                fn_del_file "${cann_uninstall_same}"
                fn_del_config "${form_path}"
                fn_del_dir "${form_path}/combo_script"
                print "INFO" "The configuration file is deleted successfully."
                return
            else
                print "ERROR" "The script combo_script/uninstall.sh cannot be operated."
                exit 2
            fi
        else
            print "ERROR" "${cann_uninstall_same} file not found"
            exit 1
        fi
    fi
    if [ -f ${cann_uninstall_same} ] && [ -f ${cann_uninstall_diff} ]; then
        local ret=0
        num=$(grep "^uninstall_package " "${cann_uninstall_same}" | wc -l)
        grep "^uninstall_package \"combo_script" "${cann_uninstall_same}" 1>/dev/null 2>&1
        if [ $? -eq 0 ] && [ ${num} -eq 1 ]; then
            record_change=n
            if [ ! -w ${form_path_same} ]; then
                chmod u+w ${form_path_same}
                record_change=y
            fi
            fn_del_file "${cann_uninstall_same}"
            fn_del_file "${form_path_same}/runtime"
            fn_del_config "${form_path_same}"
            fn_del_dir "${form_path_same}" "true"
            if [ -d ${form_path_same} ] && [ ${record_change} = y ]; then
                chmod u-w ${form_path_same}
            fi
            let 'ret+=1'
        fi
        num=$(grep "^uninstall_package " "${cann_uninstall_diff}" | wc -l)
        grep "^uninstall_package \"../../combo_script" "${cann_uninstall_diff}" 1>/dev/null 2>&1
        if [ $? -eq 0 ] && [ ${num} -eq 1 ]; then
            record_change=n
            if [ ! -w ${form_path_diff} ]; then
                chmod u+w ${form_path_diff}
                record_change=y
            fi
            fn_del_file "${cann_uninstall_diff}"
            fn_del_dir "${form_path_diff}/hetero-arch-scripts" "true"
            fn_del_config "${form_path_diff}"
            fn_del_file "${form_path_diff}/runtime"
            fn_del_dir "${form_path_diff}" "true"
            if [ -d ${form_path_diff} ] && [ ${record_change} = y ]; then
                chmod u-w ${form_path_diff}
            fi
            let 'ret+=1'
        fi
        if [ ${ret} -eq 0 ]; then
            print "ERROR" "The script combo_script/uninstall.sh cannot be operated."
            exit 3
        elif [ ${ret} -eq 2 ]; then
            fn_del_dir "${form_path}/combo_script"
        fi
        print "INFO" "The configuration file is deleted successfully."
    elif [ ! -f ${cann_uninstall_same} ] && [ ! -f ${cann_uninstall_diff} ];then
        print "ERROR" "${cann_uninstall_same} and ${cann_uninstall_diff} file not found"
        exit 1
    else
        if [ -f ${cann_uninstall_same} ]; then
            num=$(grep "^uninstall_package " "${cann_uninstall_same}" | wc -l)
            grep "^uninstall_package \"combo_script" "${cann_uninstall_same}" 1>/dev/null 2>&1
            if [ $? -eq 0 ] && [ ${num} -eq 1 ]; then
                if [ -d ${form_path_same} ]; then
                    record_change=n
                    if [ ! -w ${form_path_same} ]; then
                        chmod u+w ${form_path_same}
                        record_change=y
                    fi
                    fn_del_config "${form_path_same}"
                    fn_del_file "${form_path_same}/runtime"
                    fn_del_dir "${form_path_same}" "true"
                    if [ -d ${form_path_same} ] && [ ${record_change} = y ]; then
                        chmod u-w ${form_path_same}
                    fi
                fi
                if [ -d ${form_path_diff} ]; then
                    record_change=n
                    if [ ! -w ${form_path_diff} ]; then
                        chmod u+w ${form_path_diff}
                        record_change=y
                    fi
                    fn_del_config "${form_path_diff}"
                    fn_del_file "${form_path_diff}/runtime"
                    fn_del_dir "${form_path_diff}" "true"
                    if [ -d ${form_path_diff} ] && [ ${record_change} = y ]; then
                        chmod u-w ${form_path_diff}
                    fi
                fi
                fn_del_file "${cann_uninstall_same}"
                fn_del_dir "${form_path}/combo_script"
            else
                print "ERROR" "The script combo_script/uninstall.sh cannot be operated."
                exit 4
            fi
        else
            num=$(grep "^uninstall_package " "${cann_uninstall_diff}" | wc -l)
            grep "^uninstall_package \"../../combo_script" "${cann_uninstall_diff}" 1>/dev/null 2>&1
            if [ $? -eq 0 ] && [ ${num} -eq 1 ]; then
                record_change=n
                if [ ! -w ${form_path_diff} ]; then
                    chmod u+w ${form_path_diff}
                    record_change=y
                fi
                fn_del_file "${cann_uninstall_diff}"
                fn_del_dir "${form_path_diff}/hetero-arch-scripts" "true"
                fn_del_config "${form_path_diff}"
                fn_del_file "${form_path_diff}/runtime"
                fn_del_dir "${form_path_diff}" "true"
                fn_del_dir "${form_path}/combo_script"
                if [ -d ${form_path_diff} ] && [ ${record_change} = y ]; then
                    chmod u-w ${form_path_diff}
                fi
            else
                print "ERROR" "The script combo_script/uninstall.sh cannot be operated."
                exit 5
            fi
        fi
        print "INFO" "The configuration file is deleted successfully."
    fi
}

function fn_del_config() {
    local temp_path=${1}
    fn_del_file "${temp_path}/ascend_${PACKAGE_SHORT_NAME}_install.info"
    fn_del_file "${temp_path}/install.conf"
    chmod 700 "${temp_path}/script"
    fn_del_file "${temp_path}/script/uninstall.sh"
    fn_del_file "${temp_path}/script/set_env.sh"
    fn_del_dir "${temp_path}/script" "true"
}

# 更新latest下目录
function latest_dir_upgrade() {
    remove_dir_invalid_soft "${form_path}"
    remove_dir_invalid_soft "${version_path}/latest/aarch64-linux"
    remove_dir_invalid_soft "${version_path}/latest/x86_64-linux"
    remove_dir_invalid_soft "${version_path}/latest/sw_64-linux"
    remove_dir_invalid_soft "${version_path}/latest"
    remove_dir_invalid_soft "${version_path}"
    local upgrade_path=""
    if [ -h "${version_path}/latest/runtime" ] && [ -e "${version_path}/latest/runtime" ]; then
        upgrade_path=$(dirname $(readlink -f "${version_path}/latest/runtime"))
    elif [ -h "${version_path}/latest/fwkplugin" ] && [ -e "${version_path}/latest/fwkplugin" ]; then
        upgrade_path=$(dirname $(readlink -f "${version_path}/latest/fwkplugin"))
    else
        fn_del_dir "${version_path}/latest" "true"
        fn_del_file "${version_path}/set_env.sh"
        fn_del_dir "${version_path}" "true"
        find "${version_path}/latest" -maxdepth 1 -type l -name "arm64-linux" -exec rm {} \;
        return
    fi
    local upgrade_package_version=$(basename ${upgrade_path})
    local version_link=$(echo ${upgrade_package_version} | cut -f1-2 -d".")
    if [ x"${PACKAGE_VERSION_FORM}" == x"" ]; then
        create_config_soft "${version_path}/latest" "${upgrade_path}" "../${upgrade_package_version}"
    else
        if [ ! -h "${version_path}/latest/x86_64-linux" ] && [ -d "${version_path}/latest/x86_64-linux" ]; then
            create_config_soft "${version_path}/latest/x86_64-linux" "${upgrade_path}/x86_64-linux" "../../${upgrade_package_version}/x86_64-linux"
        fi
        if [ ! -h "${version_path}/latest/aarch64-linux" ] && [ -d "${version_path}/latest/aarch64-linux" ]; then
            create_config_soft "${version_path}/latest/aarch64-linux" "${upgrade_path}/aarch64-linux" "../../${upgrade_package_version}/aarch64-linux"
        fi
    fi
    create_form_soft "${upgrade_package_version}"
    create_file_soft "./${upgrade_package_version}" "${version_path}/${version_link}"
    print "INFO" "The soft link is updated successfully."
}

# 删除目录下无效软链接
function remove_dir_invalid_soft() {
    local temp_path=${1}
    if [ ! -d ${temp_path} ]; then
        return
    fi
    local record_change=n
    if [ ! -w ${temp_path} ]; then
        chmod u+w ${temp_path}
        record_change=y
    fi
    local link_files=$(ls -l ${temp_path} 2>/dev/null | grep "^l" | awk '{print $9}' | sed "s#^#${temp_path}/#g")
    link_files=$(echo ${link_files} | tr "\n" " ")
    for link_file in ${link_files}; do
        if [ ! -e $link_file ]; then
            print "INFO" "rm soft link ${link_file}"
            fn_del_file ${link_file}
        fi
    done
    fn_del_dir "${temp_path}" "true"
    if [ -d ${temp_path} ] && [ ${record_change} = y ]; then
        chmod u-w ${temp_path}
    fi
}

# 创建配置文件软链接
function create_config_soft() {
    local target_dir=${1}
    local origin_dir=${2}
    local soft_dir=${3}
    local soft_obj=("ascend_${PACKAGE_SHORT_NAME}_install.info" "install.conf" "script")
    local record_change=n
    if [ ! -w ${target_dir} ]; then
        chmod u+w ${target_dir}
        record_change=y
    fi
    for file_var in ${soft_obj[@]}; do
        if [ -e "${origin_dir}/${file_var}" ]; then
            create_file_soft "${soft_dir}/${file_var}" "${target_dir}/${file_var}"
            print "INFO" "The soft link ${target_dir}/${file_var} is created successfully."
        fi
    done
    if [ x"${PACKAGE_VERSION_FORM}" != x"" ] && [ -h "${origin_dir}/runtime" ]; then
        create_file_soft "${soft_dir}/runtime" "${target_dir}/runtime"
    fi
    if [ ${record_change} = y ]; then
        chmod u-w ${target_dir}
    fi
}

function create_form_soft() {
    local temp_package_version=${1}
    local form_soft=("acllib_linux.arm64" "runtime_linux.arm64" "opp_linux.arm64" "acllib_linux.x86_64" "runtime_linux.x86_64" "opp_linux.x86_64")
    for soft_var in ${form_soft[@]}; do
        if [ -h "${version_path}/${temp_package_version}/${soft_var}" ]; then
            create_file_soft "../${temp_package_version}/${soft_var}" "${version_path}/latest/${soft_var}"
        fi
    done
    if [ -h "${version_path}/${temp_package_version}/arm64-linux" ]; then
        create_file_soft "aarch64-linux" "${version_path}/latest/arm64-linux"
    fi
    # 卸载兼容旧版本的 mindstudio-msprof 创建
    if [ -d "${version_path}/${temp_package_version}/mindstudio-msprof" ] && [ ! -h "${version_path}/latest/mindstudio-msprof" ] ; then
        create_file_soft "../${temp_package_version}/mindstudio-msprof" "${version_path}/latest/mindstudio-msprof"
    fi
}

# 创建软链接
function create_file_soft() {
    local origin_obj=${1}
    local target_obj=${2}
    [ -h "${target_obj}" ] && fn_del_file "${target_obj}"
    if [ -e "${target_obj}" ]; then
        print "WARNING" "The target ${target_obj} exists, and the soft link cannot be established."
    else
        ln -sf "${origin_obj}" "${target_obj}"
        print "INFO" "The soft link ${target_obj} is created successfully."
    fi
}

# 判断是否移除ascend_cann_install.info路径配置文件
function remove_config_info() {
    # 路径使用配置文件
    if [ -f ${CANN_INSTALL_INFO} ]; then
        # 参数配置
        local param=$(cat "${CANN_INSTALL_INFO}" | grep -w "Install_Path" | cut -d"=" -f2 | sed "s/ //g")
        if [ ! -n "${param}" ]; then
            print "WARNING" "${CANN_INSTALL_INFO} Install_Path content is missing."
            print "INFO" "${CANN_INSTALL_INFO} uninstall success"
            return
        fi
        local Install_Path=${param}
        local tmp_files=$(ls ${Install_Path}/*/*/*/ascend_*_install.info 2>/dev/null)
        if [ x"${tmp_files}" = "x" ]; then
            tmp_files=$(ls ${Install_Path}/*/*/ascend_*_install.info 2>/dev/null)
        fi
        local cann_package=("nnrt" "nnae" "toolkit" "tfplugin")
        for package in ${cann_package[@]}; do
            if [[ "${tmp_files}" =~ "$package" ]]; then
                return
            fi
        done
        fn_del_file ${CANN_INSTALL_INFO}
        print "INFO" "${CANN_INSTALL_INFO} uninstall success"
    fi
}

function del_pkg() {
    if [ -f ${cann_uninstall_same} ]; then
        local infos=$(grep "^uninstall_package " ${cann_uninstall_same} | cut -d " " -f2 | sed "s/\"//g")
        for info in $infos; do
            if [ "$info" != "combo_script" ] && [ -f "${form_path}/${info}/uninstall.sh" ]; then
                ${form_path}/${info}/uninstall.sh
            fi
        done      
    fi
}

# 执行删除子run包
function deal_uninstall() {
    # 删除冷补丁备份目录
    remove_spc_dir
    # 删除组合包配置文件
    del_pkg
    del_config_file
    # latest 目录下软链接更新
    latest_dir_upgrade
    # 移除路径配置文件
    remove_config_info
}

# 程序开始
function main() {
    deal_uninstall
}

main
