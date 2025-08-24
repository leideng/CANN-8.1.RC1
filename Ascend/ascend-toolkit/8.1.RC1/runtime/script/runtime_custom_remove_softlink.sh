#!/bin/sh
# Perform custom remove softlink script for runtime package
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
    local filelist="$install_path/$version_dir/runtime/script/filelist.csv"
    grep -o "CommonLib1,[^/]\+/lib64/stub/linux/$arch_name/[^,]\+" $filelist 2>/dev/null | xargs --no-run-if-empty -n 1 basename
}

common_stub_in_use() {
    local scene_file="$install_path/$version_dir/compiler/scene.info"
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
    [ -L "x86_64" ] && rm -rf "x86_64"
    [ -L "aarch64" ] && rm -rf "aarch64"
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
    local arch_name="$(get_arch_name $install_path/$version_dir/runtime)"
    local arch_linux_path="$install_path/$latest_dir/$arch_name-linux"
    if [ ! -e "$arch_linux_path" ] || [ -L "$arch_linux_path" ]; then
        return
    fi
    local ref_dir="$install_path/$version_dir/runtime/lib64/stub/linux/$arch_name"
    remove_stub_softlink "$ref_dir" "$arch_linux_path/devlib"
    remove_stub_softlink "$ref_dir" "$arch_linux_path/lib64/stub"
    recreate_common_stub_softlink "$arch_name" "$arch_linux_path/devlib"
}

remove_helper_softlink() {
    if [ -L "$install_path/$latest_dir/modeldeployer" ]; then
        rm -rf "$install_path/$latest_dir/modeldeployer"
    fi
}

rm -rf "$install_path/$latest_dir/acllib"
do_remove_stub_softlink
remove_helper_softlink
