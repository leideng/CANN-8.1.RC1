#!/bin/sh
# Perform custom_install script for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

sourcedir="$PWD/runtime"
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"

. "${common_func_path}"

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
hetero_arch="n"

while true; do
    case "$1" in
    --install-path=*)
        pkg_install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
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
    shift
    if [ "$log_type" = "INFO" -o "$log_type" = "WARNING" -o "$log_type" = "ERROR" ]; then
        echo -e "[Runtime] [$cur_date] [$log_type]: $*"
    else
        echo "[Runtime] [$cur_date] [$log_type]: $*" 1> /dev/null
    fi
    echo "[Runtime] [$cur_date] [$log_type]: $*" >> "$logfile"
}

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

custom_install() {
    if [ -z "$common_parse_dir/runtime" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:runtime directory is empty"
        exit 1
    elif [ "$hetero_arch" != "y" ]; then
        local arch_name="$(get_arch_name $common_parse_dir/runtime)"
        create_stub_softlink "$common_parse_dir/runtime/lib64/stub" "linux/$arch_name"
        create_stub_softlink "$common_parse_dir/$arch_name-linux/devlib" "linux/$arch_name"
        create_stub_softlink "$common_parse_dir/$arch_name-linux/lib64/stub" "linux/$arch_name"
    else
        local arch_name="$(get_arch_name $common_parse_dir/runtime)"
        create_stub_softlink "$common_parse_dir/runtime/lib64/stub" "linux/$arch_name"
        create_stub_softlink "$common_parse_dir/../devlib" "linux/$arch_name"
        create_stub_softlink "$common_parse_dir/../lib64/stub" "linux/$arch_name"
    fi
    return 0
}

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
