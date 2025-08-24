#!/bin/bash
# Perform install/upgrade/uninstall for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

curpath=$(dirname $(readlink -f $0))

# get input params
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

custom_install() {

    return 0
}

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0