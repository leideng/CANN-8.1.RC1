#!/bin/bash

SHELL_DIR=$(cd "$(dirname "$0")" || exit; pwd)

source ${SHELL_DIR}/common.sh

init_log

if [ $# -lt 2 ]; then
    exit 1
fi

oper_option=$1
shift 1

if [ $oper_option == "--uninstall" ]; then
    uninstall_latest $@
    exit $?
fi

if [ $oper_option == "--install" ]; then
    install_latest $@
    exit $?
fi

exit 0
