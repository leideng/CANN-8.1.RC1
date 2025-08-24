#!/bin/bash

SHELL_DIR=$(cd "$(dirname "$0")" || exit; pwd)

print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Toolkit] [${cur_date}] [$1]: $content"
}

function check_tool_version() {
    local _min_version=$1
    local _installed_version=$2

    local _min_version_arr=`echo ${_min_version} | tr "." " "`
    for _min_version_num in ${_min_version_arr[@]}; do
        _installed_version_num=`echo ${_installed_version} | cut -d"." -f1`
        _installed_version=`echo ${_installed_version} | cut -d"." -f2-`
        if [ "-${_installed_version_num}" == "-" ]; then
            return 0
        fi
        if [ ${_installed_version_num} -gt ${_min_version_num} ]; then
            return 0
        fi
        if [ ${_installed_version_num} -lt ${_min_version_num} ]; then
            return 1
        fi
    done
    return 0
}

function check_python_version() {
    local _min_version=3.7.5
    local _installed_version=`python3 --version 2>&1 | cut -d" " -f2`

    check_tool_version ${_min_version} ${_installed_version}
    if [ $? -ne 0 ]; then
        print_log "WARNING" "python version ${_installed_version} low than ${_min_version}."
    fi
}

function check_python_module_version() {
    local _module=$1

    if [ ${_module} == "protobuf" ]; then
        python3 -c "import google.protobuf" >/dev/null 2>&1
    else
        python3 -c "import ${_module}" >/dev/null 2>&1
    fi
    if [ $? -ne 0 ]; then
        print_log "WARNING" "python module ${_module} doesn't exist."
        return
    fi

    [ $# -eq 1 ] && return
    local _min_version=$2

    local _installed_version=`pip3 list 2>&1 | grep "^${_module} " | cut -d" " -f2- | grep -Eo "[0-9.]+"`
    if [ "-${_installed_version}" == "-" ]; then
        echo "search $module version failed."
        return
    fi
    check_tool_version ${_min_version} ${_installed_version}
    if [ $? -ne 0 ]; then
        print_log "WARNING" "python module ${_module} version ${_installed_version} low than ${_min_version}."
    fi
}

check_python_version
check_python_module_version sqlite3

feature_type=$1

install_info="${SHELL_DIR}/../ascend_install.info"
if [ -z "${feature_type}" ] && [ -f "${install_info}" ]; then
    feature_type=`cat "${install_info}" | grep "Feature_Type"`
fi
if [ -z "${feature_type}" ]; then
    exit 0
fi
feature_type_matched=`echo ${feature_type} | grep -Eo "op|all|model"`
if [ "-${feature_type_matched}" != "-" ]; then
    check_python_module_version numpy 1.13.3
fi
feature_type_matched=`echo ${feature_type} | grep -Eo "op|all"`
if [ "-${feature_type_matched}" != "-" ]; then
    check_python_module_version scipy 1.4.1
    check_python_module_version psutil 5.7.0
    check_python_module_version protobuf 3.13.0
fi
