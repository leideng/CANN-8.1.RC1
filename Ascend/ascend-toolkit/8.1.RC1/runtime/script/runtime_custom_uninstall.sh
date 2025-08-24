#!/bin/sh
# Perform custom uninstall script for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))

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
    local log_format="[Runtime] [$cur_date] [$log_type]: $log_msg"
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
    local filelist="$common_parse_dir/runtime/script/filelist.csv"
    grep -o "CommonLib1,[^/]\+/lib64/stub/linux/$arch_name/[^,]\+" $filelist 2>/dev/null | xargs --no-run-if-empty -n 1 basename
}

common_stub_in_use() {
    local scene_file="$common_parse_dir/compiler/scene.info"
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
        cd $stub_dir && [ "$(common_stub_in_use)" = "true" ] && [ -f "linux/$arch_name/$stub" ] && \
            chmod u+w . && ln -sf "linux/$arch_name/$stub" "$stub"
    done
    cd $pwdbak
}

custom_uninstall() {
    if [ -z "$common_parse_dir/runtime" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:runtime directory is empty"
        exit 1
    elif [ "$hetero_arch" != "y" ]; then
        local arch_name="$(get_arch_name $common_parse_dir/runtime)"
        local ref_dir="$common_parse_dir/runtime/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/runtime/lib64/stub"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/$arch_name-linux/devlib"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/$arch_name-linux/lib64/stub"
        recreate_common_stub_softlink "$arch_name" "$common_parse_dir/$arch_name-linux/devlib"
    else
        local arch_name="$(get_arch_name $common_parse_dir/runtime)"
        local ref_dir="$common_parse_dir/runtime/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/runtime/lib64/stub"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/../devlib"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/../lib64/stub"
        recreate_common_stub_softlink "$arch_name" "$common_parse_dir/../devlib"
    fi
    return 0
}

custom_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
