#!/bin/bash
# Perform uninstall for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))
common_parse_dir=""
stage=""
is_quiet="n"

while true; do
    case "$1" in
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
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

custom_uninstall() {
    return 0
}

custom_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0