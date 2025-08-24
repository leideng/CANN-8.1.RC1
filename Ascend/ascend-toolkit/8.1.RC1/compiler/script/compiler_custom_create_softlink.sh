#!/bin/bash
# Perform custom create softlink script for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"

. "${common_func_path}"

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

create_stub_softlink() {
    local stub_dir="$1"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local arch_name="$2"
    local pwdbak="$(pwd)"
    cd $stub_dir && [ -d "$arch_name" ] && for so_file in $(find "$arch_name" -type f -o -type l); do
        ln -sf "$so_file" "$(basename $so_file)"
    done
    cd $pwdbak
}

do_create_stub_softlink() {
    local arch_name="$(get_arch_name $install_path/$version_dir/compiler)"
    local arch_linux_path="$install_path/$latest_dir/$arch_name-linux"
    if [ ! -e "$arch_linux_path" ] || [ -L "$arch_linux_path" ]; then
        return
    fi
    create_stub_softlink "$arch_linux_path/devlib" "linux/$arch_name"
}

ln -srfn "$install_path/$version_dir/atc" "$install_path/$latest_dir/atc"
ln -srfn "$install_path/$version_dir/fwkacllib" "$install_path/$latest_dir/fwkacllib"
do_create_stub_softlink

create_auto_tune_soft_link() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _package_name="$3"
    local _package_egg_name="$4"
    if [ -d "${_src_dir}/${_package_name}" ]; then
        mkdir -p "${_dst_dir}/${_package_egg_name}/${_package_name}"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}/${_package_name}"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}/auto_tune_main"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/auto_tune_main"
    fi
}

create_rl_soft_link() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _package_name="$3"
    local _package_egg_name="$4"
    if [ -d "${_src_dir}/${_package_name}" ]; then
        mkdir -p "${_dst_dir}/${_package_egg_name}"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}"
    fi
}

python_dir_chmod_set() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    if [ $(id -u) -eq 0 ]; then
        chmod 755 "$dir" > /dev/null 2>&1
    else
        chmod 750 "$dir" > /dev/null 2>&1
    fi
}

python_dir_chmod_reset() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    chmod u+w "$dir" > /dev/null 2>&1
}

WHL_INSTALL_DIR_PATH="$install_path/$version_dir/python/site-packages"
WHL_SOFTLINK_INSTALL_DIR_PATH="$install_path/$latest_dir/python/site-packages"
PYTHON_AUTO_TUNE_NAME="auto_tune"
PYTHON_SEARCH_NAME="schedule_search"
PYTHON_AUTO_TUNE_EGG="auto_tune.egg"
PYTHON_SCH_SEARCH_EGG="schedule_search.egg"
mkdir -p "$WHL_SOFTLINK_INSTALL_DIR_PATH"
python_dir_chmod_reset "$WHL_SOFTLINK_INSTALL_DIR_PATH"

create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "te"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "te_fusion"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "tbe"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "te-*.dist-info"

create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "opc_tool"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "opc_tool-*.dist-info"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "op_compile_tool"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "op_compile_tool-*.dist-info"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "dataflow"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "dataflow-*.dist-info"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "llm_datadist"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "llm_datadist-*.dist-info"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "repository_manager"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "tik_tune"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "auto_tune"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "auto_deploy_utils"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "auto_tune-*.dist-info"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "optune_utils"

create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "schedule_search"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "auto_search"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "schedule_search-*.dist-info"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "autofuse"

create_auto_tune_soft_link "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "${PYTHON_AUTO_TUNE_NAME}" "${PYTHON_AUTO_TUNE_EGG}"
create_rl_soft_link "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "${PYTHON_SEARCH_NAME}" "${PYTHON_SCH_SEARCH_EGG}"

python_dir_chmod_set "$WHL_SOFTLINK_INSTALL_DIR_PATH"
python_dir_chmod_set "$(dirname $WHL_SOFTLINK_INSTALL_DIR_PATH)"
