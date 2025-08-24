#!/bin/sh
# Perform custom create softlink script for runtime package
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
    [ -d "linux/x86_64" ] && ln -snf "linux/x86_64" "x86_64"
    [ -d "linux/aarch64" ] && ln -snf "linux/aarch64" "aarch64"
    cd $pwdbak
}

do_create_stub_softlink() {
    local arch_name="$(get_arch_name $install_path/$version_dir/runtime)"
    local arch_linux_path="$install_path/$latest_dir/$arch_name-linux"
    if [ ! -e "$arch_linux_path" ] || [ -L "$arch_linux_path" ]; then
        return
    fi
    create_stub_softlink "$arch_linux_path/devlib" "linux/$arch_name"
    create_stub_softlink "$arch_linux_path/lib64/stub" "linux/$arch_name"
}

create_helper_softlink() {
    if [ -d "$install_path/$version_dir/hsheadfwk" ] || [ -d "$install_path/$version_dir/hsdevre" ]; then
        ln -srfn "$install_path/$version_dir/runtime" "$install_path/$latest_dir/modeldeployer"
    fi
}

ln -srfn "$install_path/$version_dir/runtime" "$install_path/$latest_dir/acllib"
do_create_stub_softlink
create_helper_softlink
