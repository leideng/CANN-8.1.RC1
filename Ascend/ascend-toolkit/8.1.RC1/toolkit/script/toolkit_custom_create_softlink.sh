#!/bin/bash
SHELL_DIR=$(cd "$(dirname "$0")" || exit;pwd)
COMMON_SHELL_PATH="$SHELL_DIR/common.sh"
COMMON_INC_PATH="$SHELL_DIR/common_func.inc"
FILELIST_PATH="$SHELL_DIR/filelist.csv"
VERSION_INFO="$SHELL_DIR/../version.info"
PACKAGE=toolkit
USERNAME=$(id -un)
USERGROUP=$(groups | cut -d" " -f1)

source "${COMMON_SHELL_PATH}"
source "${COMMON_INC_PATH}"

createSoftLink()
{
    local _src_dir="$1"
    local _dst_dir="$2"
    local _name="$3"

    [ ! -d "$_src_dir" -o ! -d "$_dst_dir" ] && return
    [ ! -f "$_src_dir/$_name" -a ! -d "$_src_dir/$_name" ] && return

    if [ -L "$_dst_dir/$_name" ]; then
        rm -rf "$_dst_dir/$_name"
    fi

    ln -s "$_src_dir/$_name" "$_dst_dir/$_name"
}

createPythonSoftLink()
{
    local _install_path=$1
    local _version_dir=$2
    local _latest_dir=$3
    local _src_dir="$_install_path/$_version_dir/python/site-packages"
    local _dst_dir="$_install_path/$_latest_dir/python/site-packages"

    [ -z "$_version_dir" ] && return
    [ ! -d "$_src_dir" ] && return

    if [ ! -d "$_install_path/$_latest_dir/python" ]; then
        createFolder "$_install_path/$_latest_dir/python" $USERNAME:$USERGROUP 750
        [ $? -ne 0 ] && return
    fi
    if [ ! -d "$_install_path/$_latest_dir/python/site-packages/" ]; then
        createFolder "$_install_path/$_latest_dir/python/site-packages/" $USERNAME:$USERGROUP 750
        [ $? -ne 0 ] && return
    fi

    createSoftLink "$_src_dir" "$_dst_dir" "bin"
    createSoftLink "$_src_dir" "$_dst_dir" "hccl_parser"
    createSoftLink "$_src_dir" "$_dst_dir" "hccl_parser-0.1.dist-info"
    createSoftLink "$_src_dir" "$_dst_dir" "op_gen"
    createSoftLink "$_src_dir" "$_dst_dir" "op_gen-0.1.dist-info"
    createSoftLink "$_src_dir" "$_dst_dir" "op_test_frame"
    createSoftLink "$_src_dir" "$_dst_dir" "op_test_frame-0.1.dist-info"
    createSoftLink "$_src_dir" "$_dst_dir" "msobjdump"
    createSoftLink "$_src_dir" "$_dst_dir" "msobjdump-0.1.0.dist-info"
    createSoftLink "$_src_dir" "$_dst_dir" "show_kernel_debug_data"
    createSoftLink "$_src_dir" "$_dst_dir" "show_kernel_debug_data-0.1.0.dist-info"
}

createToolSoftLink()
{
    local _install_path=$1
    local _version_dir=$2
    local _latest_dir=$3
    local _src_dir="$_install_path/$_version_dir/tools"
    local _dst_dir="$_install_path/$_latest_dir/tools"

    [ -z "$_version_dir" ] && return
    [ ! -d "$_src_dir" ] && return

    [ ! -f "$FILELIST_PATH" ] && return
    local _tool_files=$(cat "$FILELIST_PATH" | cut -d',' -f4 | grep "^tools/[^/]\+$" | cut -d"/" -f2 | sort | uniq)
    for tool_file in ${_tool_files[@]}; do
        [ -L "$_src_dir/$tool_file" ] && continue
        createSoftLink "$_src_dir" "$_dst_dir" "$tool_file"
    done
}

createCanndevSoft() {
    local _install_path=$1
    local _version_dir=$2
    local _latest_dir=$3
    local _src_dir="$_install_path/$_version_dir/python/site-packages/bin"
    local _dst_dir="$_install_path/$_latest_dir/bin"
    
    [ -z "$_version_dir" ] && return
    [ ! -d "$_src_dir" ] && return
    [ ! -d "$_dst_dir" ] && mkdir -p "$_dst_dir"  
    
    createSoftLink "$_src_dir" "$_dst_dir" "msopgen"
    createSoftLink "$_src_dir" "$_dst_dir" "msopst"
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

is_multi_version_pkg "is_multi_version" "${VERSION_INFO}"
[ ! "$is_multi_version" = "true" ] && exit 0

createPythonSoftLink "$install_path" "$version_dir" "$latest_dir"
createToolSoftLink "$install_path" "$version_dir" "$latest_dir"
createCanndevSoft "$install_path" "$version_dir" "$latest_dir"

exit 0

