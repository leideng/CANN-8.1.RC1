#!/bin/bash
# Perform custom remove softlink script for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
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

get_arch_name() {
    local pkg_dir="$1"
    local scene_file="$pkg_dir/scene.info"
    grep '^arch=' $scene_file | cut -d"=" -f2
}

get_common_stubs() {
    local arch_name="$1"
    local filelist="$install_path/$version_dir/compiler/script/filelist.csv"
    grep -o "CommonLib1,[^/]\+/lib64/stub/linux/$arch_name/[^,]\+" $filelist 2>/dev/null | xargs --no-run-if-empty -n 1 basename
}

common_stub_in_use() {
    local scene_file="$install_path/$version_dir/runtime/scene.info"
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
        cd $stub_dir && [ "$(common_stub_in_use)" = "true" ] && [ -L "linux/$arch_name/$stub" ] && \
            chmod u+w . && ln -srf "$(realpath linux/$arch_name/$stub)" "$stub"
    done
    cd $pwdbak
}

do_remove_stub_softlink() {
    local arch_name="$(get_arch_name $install_path/$version_dir/compiler)"
    local arch_linux_path="$install_path/$latest_dir/$arch_name-linux"
    if [ ! -e "$arch_linux_path" ] || [ -L "$arch_linux_path" ]; then
        return
    fi
    local ref_dir="$install_path/$version_dir/compiler/lib64/stub/linux/$arch_name"
    remove_stub_softlink "$ref_dir" "$arch_linux_path/devlib"
    recreate_common_stub_softlink "$arch_name" "$arch_linux_path/devlib"
}

rm -rf "$install_path/$latest_dir/atc"
rm -rf "$install_path/$latest_dir/fwkacllib"
do_remove_stub_softlink

remove_softlink() {
    rm -rf $WHL_SOFTLINK_INSTALL_DIR_PATH/$1 > /dev/null 2>&1
}

remove_empty_dir() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        local is_empty=$(ls "${_path}" | wc -l)
        if [ "$is_empty" -eq 0 ]; then
            prev_path=$(dirname "${_path}")
            chmod +w "${prev_path}" > /dev/null 2>&1
            rm -rf "${_path}" > /dev/null 2>&1
        fi
    fi
}

python_dir_chmod_set() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    chmod u+w "$dir" > /dev/null 2>&1
}

WHL_SOFTLINK_INSTALL_DIR_PATH="$install_path/$latest_dir/python/site-packages"
PYTHON_AUTO_TUNE_EGG="auto_tune.egg"
PYTHON_AUTO_TUNE_MAIN="auto_tune_main"
PYTHON_SCH_SEARCH_EGG="schedule_search.egg"

python_dir_chmod_set "$WHL_SOFTLINK_INSTALL_DIR_PATH"

remove_softlink "te"
remove_softlink "te_fusion"
remove_softlink "tbe"
remove_softlink "te-*.dist-info"

remove_softlink "opc_tool"
remove_softlink "opc_tool-*.dist-info"
remove_softlink "op_compile_tool"
remove_softlink "op_compile_tool-*.dist-info"
remove_softlink "dataflow"
remove_softlink "dataflow-*.dist-info"
remove_softlink "llm_datadist"
remove_softlink "llm_datadist-*.dist-info"
remove_softlink "repository_manager"
remove_softlink "tik_tune"
remove_softlink "auto_tune"
remove_softlink "auto_tune-*.dist-info"
remove_softlink "auto_deploy_utils"
remove_softlink "optune_utils"

remove_softlink "auto_search"
remove_softlink "schedule_search"
remove_softlink "schedule_search-*.dist-info"
remove_softlink "autofuse"

remove_softlink "$PYTHON_AUTO_TUNE_EGG"
remove_softlink "$PYTHON_AUTO_TUNE_MAIN"
remove_softlink "$PYTHON_SCH_SEARCH_EGG"

remove_empty_dir "$WHL_SOFTLINK_INSTALL_DIR_PATH"
remove_empty_dir "$install_path/$latest_dir/python"
