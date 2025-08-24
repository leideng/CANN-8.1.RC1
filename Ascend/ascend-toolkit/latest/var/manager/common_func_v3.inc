#!/bin/sh
# install_common_parser.sh函数库
#----------------------------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#----------------------------------------------------------------------------

# 设置公共变量
set_global_vars() {
    local arch
    local scene_filepath="${curpath}/../scene.info"

    if [ -f "${scene_filepath}" ]; then
        get_scene_arch "arch" "${scene_filepath}"
        if [ "${arch}" != "" ]; then
            PKG_ARCH="${arch}"
        fi
    fi
}

# 获取并发进程数
get_thread_num() {
    local _outvar="$1"
    local _thread_num="$(cat /proc/cpuinfo | grep "^processor" | wc -l)"

    # 未取得CPU核数时，默认为1
    if [ "$_thread_num" = "" ] || [ "$_thread_num" = "0" ]; then
        _thread_num="1"
    fi

    # CPU核数2倍
    eval "${_outvar}=\"$((_thread_num*2))\""
}

# 初始化fifo，用于并发控制
init_fifo() {
    local _outvar="$1"
    local _thread_num="$2"
    local _tmppath_if tmp
    local _skip_msg="skip init fifo"

    # 并发数为空时，默认为1
    if [ "$_thread_num" = "" ]; then
        _thread_num="1"
        comm_log "WARNING" "thread number is empty, use ${_thread_num}."
    fi
    # 并发数<=0时，默认为1
    if [ $_thread_num -le 0 ]; then
        comm_log "WARNING" "thread number is ${_thread_num}, use 1."
        _thread_num="1"
    fi

    get_tmp_file _tmppath_if "fifo"
    if [ -e "$_tmppath_if" ]; then
        rm -rf "$_tmppath_if"
        check_ret_warning "$?" "remove old file $_tmppath_if failed, ${_skip_msg}."
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    mkfifo "$_tmppath_if"
    check_ret_warning "$?" "make fifo $_tmppath_if failed, ${_skip_msg}."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 使用文件描述符8作并发控制
    # 实现方法参考：
    #
    exec 8<> "$_tmppath_if"  # 创建文件描述符8
    check_ret_warning "$?" "open fifo $_tmppath_if file descriptor failed, ${_skip_msg}."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    rm -f "$_tmppath_if"
    check_ret_warning "$?" "remove fifo $_tmppath_if failed, ${_skip_msg}."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    for tmp in $(seq $_thread_num); do
        echo >&8
        check_ret_warning "$?" "echo fifo $_tmppath_if failed, ${_skip_msg}."
        ret="$?" && [ $ret -ne 0 ] && return $ret
    done

    eval "${_outvar}=\"${_tmppath_if}\""
}

# 根据参数执行
exec_with_param() {
    local param="$1"
    shift 1
    local exec_mode fifo_path="$PARALLEL_FIFO" tmp
    extract_1st "exec_mode" "$param"
    if [ "$exec_mode" = "concurrency" ]; then
        if [ "$fifo_path" = "none" ] || [ "$fifo_path" = "" ]; then
            "$@" &
        else
            read tmp <&8
            {
                "$@"
                ret="$?"
                echo >&8
                exit $ret
            } &
        fi
    else
        "$@"
    fi
}
