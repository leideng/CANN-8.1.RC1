#!/bin/sh
# Perform install/uninstall for latest_manager
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

CURPATH=$(dirname $(readlink -f "$0"))
USERNAME=$(id -un)
USERGROUP=$(id -gn)
common_func_path="${CURPATH}/common_func.inc"

. "$common_func_path"

INSTALL_PATH=""
IS_UPGRADE="n"

set_comm_log "Latest_manager" "$COMM_LOGFILE"

while true
do
    case "$1" in
    --install-path=*)
        INSTALL_PATH="$(echo "$1" | cut -d"=" -f2-)"
        shift
        ;;
    --upgrade)
        IS_UPGRADE="y"
        shift
        ;;
    -*)
        comm_log "ERROR" "Unsupported parameters : $1"
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

if [ "$INSTALL_PATH" = "" ]; then
    comm_log "ERROR" "--install-path parameter is required!"
    exit 1
fi

if ! sh "$CURPATH/install_common_parser.sh" --package="latest_manager" --install --username="$USERNAME" --usergroup="$USERGROUP" \
    --simple-install "full" "$INSTALL_PATH" "$CURPATH/filelist.csv" "all"; then
    comm_log "ERROR" "install failed!"
    exit 1
fi

if [ "$IS_UPGRADE" = "y" ] && ! "$INSTALL_PATH/manager.sh" "migrate_latest_data"; then
    comm_log "ERROR" "migrate latest data failed!"
    exit 1
fi
