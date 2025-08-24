#!/bin/sh

SHELL_DIR="$(dirname "${BASH_SOURCE:-$0}")"
INSTALL_PATH="$(cd "${SHELL_DIR}" && pwd)"
TOTAL_RET="0"

uninstall_package() {
    local path="$1"
    local cur_date ret

    if [ ! -d "${INSTALL_PATH}/${path}" ]; then
        cur_date=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$cur_date] [ERROR]: ${INSTALL_PATH}/${path}: No such file or directory"
        TOTAL_RET="1"
        return 1
    fi

    cd "${INSTALL_PATH}/${path}"
    ./uninstall.sh
    ret="$?" && [ ${ret} -ne 0 ] && TOTAL_RET="1"
    return ${ret}
}

if [ ! "$*" = "" ]; then
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$cur_date] [ERROR]: $*, parameter is not supported."
    exit 1
fi

uninstall_package "runtime/script"
uninstall_package "compiler/script"
uninstall_package "hccl/script"
uninstall_package "opp/test-ops/script"
uninstall_package "opp/script"
uninstall_package "toolkit/script"
uninstall_package "tools/aoe/script"
uninstall_package "tools/ncs/script"
uninstall_package "mindstudio-toolkit/script"
uninstall_package "pyACL/script"
uninstall_package "opp_kernel/script"
uninstall_package "combo_script"
exit ${TOTAL_RET}
