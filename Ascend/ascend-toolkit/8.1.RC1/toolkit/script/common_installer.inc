#!/bin/sh
#----------------------------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#----------------------------------------------------------------------------

# 通知latest管理器创建版本软链
notify_latest_manager_create_version_softlink() {
    local curpath install_path version_dir var_path

    set_comm_log "Notifier"

    curpath="$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))"
    install_path="$(readlink -f "$curpath/..")"
    version_dir="$(basename "$curpath")"
    var_path="$install_path/$LATEST_DIR/var"

    if [ ! -f "$var_path/manager.sh" ]; then
        comm_log "ERROR" "$var_path/manager.sh doesn't exist!"
        exit 2
    fi

    if ! "$var_path/manager.sh" --version-dir "$version_dir" create_version_softlink; then
        comm_log "ERROR" "create version softlink failed!"
        exit 1
    fi
    return 0
}

# 通知latest管理器删除latest软链
notify_latest_manager_remove_latest_softlink() {
    local curpath install_path var_path

    set_comm_log "Notifier"

    curpath="$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))"
    install_path="$(readlink -f "$curpath/..")"
    var_path="$install_path/$LATEST_DIR/var"

    if [ ! -f "$var_path/manager.sh" ]; then
        comm_log "ERROR" "$var_path/manager.sh doesn't exist!"
        exit 2
    fi

    if ! "$var_path/manager.sh" remove_latest_softlink; then
        comm_log "ERROR" "remove latest softlink failed!"
        exit 1
    fi
    return 0
}
