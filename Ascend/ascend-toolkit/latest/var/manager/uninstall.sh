#!/bin/sh
# Perform install/uninstall for latest_manager
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

CURPATH=$(dirname $(readlink -f "$0"))
VARPATH="$(dirname "$CURPATH")"
USERNAME=$(id -un)
USERGROUP=$(id -gn)
common_func_path="$CURPATH/common_func.inc"
manager_func_path="$CURPATH/manager_func.sh"

. "$common_func_path"
. "$manager_func_path"

set_comm_log "Latest_manager" "$COMM_LOGFILE"

IS_UPGRADE="n"

while true
do
    case "$1" in
    --upgrade)
        IS_UPGRADE="y"
        shift
        ;;
    *)
        break
        ;;
    esac
done

if ! sh "$CURPATH/install_common_parser.sh" --package="latest_manager" --uninstall --username="$USERNAME" --usergroup="$USERGROUP" \
    --simple-uninstall "full" "$VARPATH" "$CURPATH/filelist.csv" "all"; then
    comm_log "ERROR" "uninstall failed!"
    exit 1
fi

if [ "$IS_UPGRADE" = "n" ]; then
    remove_manager_refs "$VARPATH"
fi

if ! remove_dir_if_empty "$VARPATH"; then
    comm_log "ERROR" "uninstall failed!"
    exit 1
fi
