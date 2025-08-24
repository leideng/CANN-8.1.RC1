#!/bin/bash

MIN_PIP_VERSION=19
PYTHON_VERSION=3.7.5
FILE_NOT_EXIST="0x0080"

log() {
    local content cur_date
    content=$(echo "$@" | cut -d" " -f2-)
    cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    echo "[Opp] [$cur_date] [$1]: $content"
}

log "INFO" "Opp do pre check started."

log "INFO" "Check pip version."
which pip3 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log "WARNING" "\033[33mpip3 is not found.\033[0m"
fi

log "INFO" "Check python version."
curpath="$(dirname ${BASH_SOURCE:-$0})"
install_dir="$(realpath $curpath/..)"
common_interface=$(realpath $install_dir/script*/common_interface.bash)
if [ -f "$common_interface" ]; then
    . "$common_interface"
    py_version_check
fi

log "INFO" "Opp do pre check finished."