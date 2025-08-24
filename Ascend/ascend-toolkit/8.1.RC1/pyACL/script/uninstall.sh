#!/bin/bash
# 此处定义各种变量
readonly PACKAGE_SHORT_NAME="pyACL"
upgrade_flag=n

# 路径
install_path="$(dirname $(
    cd "$(dirname "$0")"
    pwd
))"
python_path="$(
    cd "$(dirname "$0")/../../python"
    pwd
)"
latest_path="$(
    cd "$(dirname "$0")/../../../latest"
    pwd
)"

config_file_path="${install_path}/ascend-${PACKAGE_SHORT_NAME}_install.info"
version_file_path="${install_path}/version.info"
scene_file_path="${install_path}/scene.info"

function print() {
    # 将关键信息打印到屏幕上
    echo "[pyACL] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
}

#安全删除文件
function rm_file_safe() {
    local file_path=$1
    # 判断变量是否为空
    if [ -n "${file_path}" ]; then
        # 判断是否是文件
        if [ -f "${file_path}" ] || [ -h "${file_path}" ]; then
            rm -f "${file_path}"
            print "INFO" "delete file ${file_path} successfully"
        else
            print "WARNING" "the file ${file_path} is not exist"
        fi
    else
        print "WARNING" "the file ${file_path} path is NULL"
    fi
}

#安全删除文件夹
function rm_dir_safe() {
    local dir_path=$1
    # 判断变量不为空且不是系统根盘
    if [ -n "${dir_path}" ] && [[ ! "${dir_path}" =~ ^/+$ ]]; then
        # 判断是否是目录
        if [ -d "${dir_path}" ]; then
            rm -rf "${dir_path}"
            print "INFO" "delete directory ${dir_path} successfully"
        else
            print "WARNING" "the directory ${dir_path} is not exist"
        fi
    else
        print "WARNING" "the directory ${dir_path} path is NULL"
    fi
}

function delete_empty_folder() {
    if [ -d "${1}" ]; then
        if [ ! "$(ls -A ${1})" ]; then
            rm_dir_safe ${1}
        fi
    fi
}

# 更改目录下文件权限实施修改
chmod_to_modify() {
    chmod 750 -R $install_path 2> /dev/null
    chmod 750 $python_path 2> /dev/null
    chmod 750 ${latest_path}/python/site-packages 2> /dev/null
}

# 移除latest软连接
function remove_latest_link() {
    if [ -L "${latest_path}/pyACL" ]; then
        rm_file_safe ${latest_path}/pyACL
    fi
    if [ -L "${latest_path}/python/site-packages/acl.so" ]; then
        rm_file_safe ${latest_path}/python/site-packages/acl.so
    fi
    if [ -L "${latest_path}/python/site-packages/acl/acl.so" ]; then
       rm_file_safe ${latest_path}/python/site-packages/acl/acl.so
    fi
}

# 创建latest python目录
function create_python_dir() {
    if [ ! -d "${latest_path}/python" ]; then
        mkdir -p ${latest_path}/python 2>/dev/null
        print "INFO" "mkdir insatll path ${latest_path}/python"
        if [ $? -ne 0 ]; then
            print "ERROR" "mkdir insatll path ${latest_path}/python permission denied"
            exit 1
        fi
        chmod 750 -R ${latest_path}/python
    fi
    if [ ! -d "${latest_path}/python/site-packages" ]; then
        mkdir -p $latest_path/python/site-packages 2>/dev/null
        print "INFO" "mkdir insatll path ${latest_path}/python/site-packages"
        if [ $? -ne 0 ]; then
            print "ERROR" "mkdir insatll path ${latest_path}/python/site-packages permission denied"
            exit 1
        fi
        chmod 750 -R ${latest_path}/python/site-packages
    fi
    if [ ! -d "${latest_path}/python/site-packages/acl" ]; then
        mkdir -p ${latest_path}/python/site-packages/acl 2>/dev/null
        print "INFO" "mkdir insatll path ${latest_path}/python/site-packages/acl"
        if [ $? -ne 0 ]; then
            print "ERROR" "mkdir insatll path ${latest_path}/python/site-packages/acl permission denied"
            exit 1
        fi
        chmod 750 -R ${latest_path}/python/site-packages/acl
    fi
}

# 添加latest软连接
function add_latest_link() {
    create_python_dir
    ln -sf ../${CURRENT_VERSION}/pyACL ${latest_path}/pyACL
    ln -sf ../../../${CURRENT_VERSION}/python/site-packages/acl.so ${latest_path}/python/site-packages/acl.so
    ln -sf ../../../../${CURRENT_VERSION}/python/site-packages/acl/acl.so ${latest_path}/python/site-packages/acl/acl.so
}

# 移除latest python目录
function remove_python_dir() {
    delete_empty_folder "${latest_path}/python/site-packages/acl"
    delete_empty_folder "${latest_path}/python/site-packages"
    delete_empty_folder "${latest_path}/python"
    delete_empty_folder "${latest_path}"
}

# latest 软链接依据runtime版本更新
function switch_to_the_previous_version() {
    PYACL_VERSION=""
    if [ -L "${latest_path}/runtime" ]; then
        runtime_path=$(dirname $(readlink -f ${latest_path}/runtime))
        CURRENT_VERSION=$(basename ${runtime_path})
        if [ -d "${runtime_path}/pyACL" ]; then
            remove_latest_link
            add_latest_link
        else
            remove_latest_link
            remove_python_dir
        fi
    else
        remove_latest_link
        remove_python_dir # 如果没有 runtime 删掉latest的python目录
    fi
}

function __remove_uninstall_package() {
    local uninstall_file=$1
    if [ -f "${uninstall_file}" ]; then
        sed -i "/uninstall_package \"pyACL\/script\"/d" "${uninstall_file}"
        if [ $? -ne 0 ]; then
            print "ERROR" "remove ${uninstall_file} uninstall_package command failed!"
            exit 1
        fi
    fi
    num=$(grep "^uninstall_package " ${uninstall_file} | wc -l)
    if [ ${num} -eq 0 ]; then
        rm -f "${uninstall_file}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            print "ERROR" "delete file: ${uninstall_file}failed, please delete it by yourself."
        fi
    fi
}

function unregist_uninstall() {
    if [ -f "${totals_vresion_path}/cann_uninstall.sh" ]; then
        chmod u+w ${totals_vresion_path}/cann_uninstall.sh
        __remove_uninstall_package "${totals_vresion_path}/cann_uninstall.sh"
        if [ -f "${totals_vresion_path}/cann_uninstall.sh" ]; then
            chmod u-w ${totals_vresion_path}/cann_uninstall.sh
        fi
    fi
}

function deal_python_dir() {
    delete_empty_folder "$python_path/site-packages/acl/"
    delete_empty_folder "$python_path/site-packages"
    delete_empty_folder "$python_path"
}

function deal_install_dir() {
    rm_file_safe ${install_path}/script/uninstall.sh
    delete_empty_folder "${install_path}/script/"
    delete_empty_folder "${install_path}"
}

function deal_uninstall() {
    rm_file_safe ${config_file_path}
    rm_file_safe ${version_file_path}
    rm_file_safe ${scene_file_path}
    rm_file_safe ${install_path}/python
    rm_file_safe "$python_path/site-packages/acl/acl.so"
    rm_file_safe "$python_path/site-packages/acl.so"
    deal_python_dir
    deal_install_dir
    if [ "${upgrade_flag}" != y ]; then
        switch_to_the_previous_version
    fi
    totals_vresion_path=${install_path%/*}
    unregist_uninstall
    delete_empty_folder "${totals_vresion_path}"
}

# 程序开始
function main() {
    upgrade_flag=$1
    chmod_to_modify
    deal_uninstall
}

main $*
