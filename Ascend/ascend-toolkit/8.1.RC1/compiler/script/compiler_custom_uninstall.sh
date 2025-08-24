#!/bin/bash
# Perform custom uninstall script for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))
unset PYTHONPATH

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
hetero_arch="n"

while true; do
    case "$1" in
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        pkg_version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --hetero-arch=*)
        hetero_arch=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

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
    echo "$log_format" >> "$logfile"
}

get_arch_name() {
    local pkg_dir="$1"
    local scene_file="$pkg_dir/scene.info"
    grep '^arch=' $scene_file | cut -d"=" -f2
}

get_common_stubs() {
    local arch_name="$1"
    local filelist="$common_parse_dir/compiler/script/filelist.csv"
    grep -o "CommonLib1,[^/]\+/lib64/stub/linux/$arch_name/[^,]\+" $filelist 2>/dev/null | xargs --no-run-if-empty -n 1 basename
}

common_stub_in_use() {
    local scene_file="$common_parse_dir/runtime/scene.info"
    if [ -f "$scene_file" ]; then
        echo true
    else
        echo false
    fi
}

remove_stub_softlink() {
    local ref_dir="$1"
    if [ ! -d "$ref_dir" ]; then
        return
    fi
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    cd $stub_dir && chmod u+w . && ls -1 "$ref_dir" | xargs --no-run-if-empty rm -rf
    cd $pwdbak
}

recreate_common_stub_softlink() {
    local arch_name="$1"
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    for stub in $(get_common_stubs "$arch_name"); do
        cd $stub_dir && [ "$(common_stub_in_use)" = "true" ] && [ -f "linux/$arch_name/$stub" ] && \
            chmod u+w . && ln -sf "linux/$arch_name/$stub" "$stub"
    done
    cd $pwdbak
}

whl_uninstall_package() {
    local _module="$1"
    local _module_apth="$2"
    if [ ! -d "${WHL_INSTALL_DIR_PATH}/${_module}" ]; then
        pip3 show "${_module}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "${_module} is not exist."
        else
            pip3 uninstall -y "${_module}" 1> /dev/null
            local ret=$?
            if [ $ret -ne 0 ]; then
                log "WARNING" "uninstall ${_module} failed, error code: $ret."
                exit 1
            else
                log "INFO" "${_module} uninstalled successfully!"
            fi
        fi
    else
        export PYTHONPATH="${_module_apth}"
        pip3 uninstall -y "${_module}" > /dev/null 2>&1
        local ret=$?
        if [ $ret -ne 0 ]; then
            log "WARNING" "uninstall ${_module} failed, error code: $ret."
            exit 1
        else
            log "INFO" "${_module} uninstalled successfully!"
        fi
    fi
}

tbe_rm_ddk_info() {
    local _file="$1"
    if [ -f "${_file}" ]; then
        chmod +w "${_file}" > /dev/null 2>&1
        rm -rf "${_file}" > /dev/null 2>&1
    fi
}

egg_uninstall_package() {
    local _install_dir="$1"
    if [ -d "${_install_dir}" ]; then
        chmod +w -R "${_install_dir}" 2> /dev/null
        chmod +w "${WHL_INSTALL_DIR_PATH}" 2> /dev/null
        rm -rf "${_install_dir}" > /dev/null 2>&1
        local ret=$?
        if [ $ret -eq 0 ]; then
            log "INFO" "${_install_dir} uninstalled successfully!"
        else
            log "WARNING" "uninstall ${_install_dir} failed, error code: $ret."
            exit 1
        fi
    fi
}

remove_auto_tune_soft_link() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        rm -rf "${_path}"
    fi
    if [ -L "${WHL_INSTALL_DIR_PATH}/auto_tune_main" ]; then
        rm -rf "${WHL_INSTALL_DIR_PATH}/auto_tune_main"
    fi
}

remove_rl_soft_link() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        rm -rf "${_path}"
    fi
}

remove_empty_dir() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        local is_empty=$(ls "${_path}" | wc -l)
        if [ "$is_empty" -ne 0 ]; then
            log "INFO" "${_path} dir is not empty."
        else
            prev_path=$(dirname "${_path}")
            chmod +w "${prev_path}" > /dev/null 2>&1
            rm -rf "${_path}" > /dev/null 2>&1
        fi
    fi
}

remove_last_license() {
    if [ -d "$WHL_INSTALL_DIR_PATH" ]; then
        if [ -f "$WHL_INSTALL_DIR_PATH/LICENSE" ]; then
            rm -rf "$WHL_INSTALL_DIR_PATH/LICENSE"
        fi
    fi
}

remove_autofuze_custom_soft_link() {
    local whl_install_dir="$1"
    if [ -L "$whl_install_dir/autofuse/dynamic" ]; then
        rm -rf "$whl_install_dir/autofuse/dynamic"
    fi
}

WHL_SOFTLINK_INSTALL_DIR_PATH="${common_parse_dir}/compiler/python/site-packages"
WHL_INSTALL_DIR_PATH="${common_parse_dir}/python/site-packages"
TE_NAME="te"
OPC_NAME="opc_tool"
OP_COMPILE="op_compile_tool"
DATAFLOW_NAME="dataflow"
LLM_DATADIST_NAME="llm_datadist"
PYTHON_AUTO_TUNE_NAME="auto_tune"
PYTHON_AUTO_DEPLOY_UTILS_NAME="auto_deploy_utils"
PYTHON_SEARCH_NAME="schedule_search"
PYTHON_AUTO_TUNE_EGG="auto_tune.egg"
PYTHON_SCH_SEARCH_EGG="schedule_search.egg"
AUTO_TUNE_SOFT_LINK_DIR="${common_parse_dir}/python/site-packages/auto_tune.egg"
SEARCH_SOFT_LINK_DIR="${common_parse_dir}/python/site-packages/schedule_search.egg"

custom_uninstall() {
    if [ -z "$common_parse_dir/compiler" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:compiler directory is empty"
        exit 1
    fi

    if [ "$hetero_arch" != "y" ]; then
        local arch_name="$(get_arch_name $common_parse_dir/compiler)"
        local ref_dir="$common_parse_dir/compiler/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/compiler/lib64/stub"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/$arch_name-linux/devlib"
        recreate_common_stub_softlink "$arch_name" "$common_parse_dir/$arch_name-linux/devlib"
    else
        local arch_name="$(get_arch_name $common_parse_dir/compiler)"
        local ref_dir="$common_parse_dir/compiler/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/compiler/lib64/stub"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/../devlib"
        recreate_common_stub_softlink "$arch_name" "$common_parse_dir/../devlib"
    fi

    if [ "$hetero_arch" != "y" ]; then
        chmod +w -R "$curpath" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/te" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/te_fusion" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/tbe" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/te-0.4.0.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/opc_tool" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/opc_tool-0.1.0.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/op_compile_tool" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/op_compile_tool-0.1.0.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/dataflow" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/dataflow-0.0.1.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/llm_datadist" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/llm_datadist-0.0.1.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/repository_manager" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/tik_tune" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/auto_tune" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/auto_tune-0.1.0.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/auto_deploy_utils" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/auto_deploy_utils-0.1.0.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/optune_utils" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/schedule_search" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/schedule_search-0.0.1.dist-info" 2> /dev/null

        log "INFO" "uninstall tbe compiler tool begin..."
        remove_autofuze_custom_soft_link "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${TE_NAME}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${OPC_NAME}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${OP_COMPILE}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${DATAFLOW_NAME}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${LLM_DATADIST_NAME}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${PYTHON_AUTO_TUNE_NAME}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${PYTHON_AUTO_DEPLOY_UTILS_NAME}" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "${PYTHON_SEARCH_NAME}" "${WHL_INSTALL_DIR_PATH}"
        if [ -f "${common_parse_dir}/compiler/ddk_info" ]; then
            tbe_rm_ddk_info "${common_parse_dir}/compiler/ddk_info"
        fi
        log "INFO" "tbe compiler tool uninstalled successfully!"
        egg_uninstall_package "${WHL_INSTALL_DIR_PATH}/${PYTHON_SCH_SEARCH_EGG}"
        egg_uninstall_package "${WHL_INSTALL_DIR_PATH}/${PYTHON_AUTO_TUNE_EGG}"
        remove_auto_tune_soft_link "${AUTO_TUNE_SOFT_LINK_DIR}"
        remove_rl_soft_link "${SEARCH_SOFT_LINK_DIR}"
    fi

    test -d "$WHL_SOFTLINK_INSTALL_DIR_PATH" && rm -rf "$WHL_SOFTLINK_INSTALL_DIR_PATH" > /dev/null 2>&1
    test -d "${common_parse_dir}/compiler/python/func2graph" && rm -rf "${common_parse_dir}/compiler/python/func2graph" > /dev/null 2>&1
    remove_empty_dir "${common_parse_dir}/compiler/python"

    if [ "$hetero_arch" != "y" ]; then
        if [ -d "${WHL_INSTALL_DIR_PATH}" ]; then
            local python_path=$(dirname "$WHL_INSTALL_DIR_PATH")
            chmod +w "${python_path}"
            remove_last_license
        fi

        remove_empty_dir "${WHL_INSTALL_DIR_PATH}"
        remove_empty_dir "${common_parse_dir}/python"
    fi

    return 0
}

custom_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
