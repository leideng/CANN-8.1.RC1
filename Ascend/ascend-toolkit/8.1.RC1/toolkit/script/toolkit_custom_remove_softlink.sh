#!/bin/bash
SHELL_DIR=$(cd "$(dirname "$0")" || exit;pwd)
COMMON_SHELL_PATH="$SHELL_DIR/common.sh"
COMMON_INC_PATH="$SHELL_DIR/common_func.inc"
FILELIST_PATH="$SHELL_DIR/filelist.csv"
VERSION_INFO="$SHELL_DIR/../version.info"
PACKAGE=toolkit

source "${COMMON_SHELL_PATH}"
source "${COMMON_INC_PATH}"

username=$(id -un)
usergroup=$(groups | cut -d" " -f1)

removeSoftLink()
{
    local _dst_dir="$1"
    local _name="$2"

    [ -L "$_dst_dir/$_name" ] && rm -rf "$_dst_dir/$_name"
}

removePythonSoftLink()
{
    local _install_path=$1
    local _latest_dir=$2
    local _dst_dir="$_install_path/$_latest_dir/python/site-packages"

    [ ! -d "$_dst_dir" ] && return

    removeSoftLink "$_dst_dir" "bin"
    removeSoftLink "$_dst_dir" "hccl_parser"
    removeSoftLink "$_dst_dir" "hccl_parser-0.1.dist-info"
    removeSoftLink "$_dst_dir" "op_gen"
    removeSoftLink "$_dst_dir" "op_gen-0.1.dist-info"
    removeSoftLink "$_dst_dir" "op_test_frame"
    removeSoftLink "$_dst_dir" "op_test_frame-0.1.dist-info"
    removeSoftLink "$_dst_dir" "msobjdump"
    removeSoftLink "$_dst_dir" "msobjdump-0.1.0.dist-info"
    removeSoftLink "$_dst_dir" "show_kernel_debug_data"
    removeSoftLink "$_dst_dir" "show_kernel_debug_data-0.1.0.dist-info"

    isDirEmpty "$_dst_dir"
    if [ $? -eq 0 ]; then
        rm -rf "$_dst_dir"
        isDirEmpty "$_install_path/$_latest_dir/python/"
        if [ $? -eq 0 ]; then
            rm -rf "$_install_path/$_latest_dir/python/"
        fi
    fi
}

removeToolSoftLink()
{
    local _install_path=$1
    local _version_dir=$2
    local _latest_dir=$3
    local _src_dir="$_install_path/$_version_dir/tools"
    local _dst_dir="$_install_path/$_latest_dir/tools"

    [ ! -d "$_src_dir" -o ! -d "$_dst_dir" ] && return

    [ ! -f "$FILELIST_PATH" ] && return
    local _tool_files=$(cat "$FILELIST_PATH" | cut -d',' -f4 | grep "^tools/[^/]\+$" | cut -d"/" -f2 | sort | uniq)
    for tool_file in ${_tool_files[@]}; do
        [ -L "$_src_dir/$tool_file" ] && continue
        removeSoftLink "$_dst_dir" "$tool_file"
    done
}

removeCanndevSoftLink() {
    local _install_path=$1
    local _latest_dir=$2
    local _dst_dir="$_install_path/$_latest_dir/bin"
    
    [ ! -d "$_dst_dir" ] && return
    
    removeSoftLink "$_dst_dir" "msopgen"
    removeSoftLink "$_dst_dir" "msopst"
}

install_path=""
version_dir=""
latest_dir=""

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        [ -z "${install_path}" ] && exit 1
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2-)
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

is_multi_version_pkg "is_multi_version" "$VERSION_INFO"
[ ! "$is_multi_version" = "true" ] && exit 0

removePythonSoftLink "$install_path" "$latest_dir"
removeToolSoftLink "$install_path" "$version_dir" "$latest_dir"
removeCanndevSoftLink "$install_path" "$latest_dir"

exit 0
