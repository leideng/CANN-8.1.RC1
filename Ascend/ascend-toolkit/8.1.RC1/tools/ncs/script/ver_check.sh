#!/bin/bash
# Perform version check for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

# Check the relationship of ncs with fwk and atc
req_ver_pathver=$1
in_checkpath=$(echo "${req_ver_pathver}" | cut -d"," -f1)
last_component=$(basename "${in_checkpath}")

# run package's files info
_CURR_PATH=$(dirname $(readlink -f $0))
_DEFAULT_INSTALL_PATH="/usr/local/Ascend"
# defaults for general user
if [[ "$(id -u)" != "0" ]]; then
    _DEFAULT_USERNAME="${_CURR_OPERATE_USER}"
    _DEFAULT_USERGROUP="${_CURR_OPERATE_GROUP}"
    _DEFAULT_INSTALL_PATH="/home/${_CURR_OPERATE_USER}/Ascend"
fi

function get_date() {
    local _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${_cur_date}"
}

function log_and_print() {
    echo "[ncs] [$(get_date)] ""$1"
}

function version_check () {
  if [ -f "${_CURR_PATH}/../../version.info" ];then
    ver_info="${_CURR_PATH}/../../version.info"
  else
    ver_info="${_DEFAULT_INSTALL_PATH}/ncs/version.info"
  fi

  req_pkg_name="${last_component}"

  req_pkg_name_path="${in_checkpath}/version.info"

  log_and_print "[INFO]: req_pkg_name is ${req_pkg_name}, req_pkg_name_path is ${req_pkg_name_path}"
  if [ -f "${ver_info}" ];then
    echo "${ver_info}" >> /dev/null 2
  else
    log_and_print "[WARNING]: The ncs version path ${ver_info} is unobtainable."
    exit 1
  fi

  if [ -f "${req_pkg_name_path}" ];then
     echo "${req_pkg_name_path}" >> /dev/null 2
  else
     log_and_print "[WARNING]: The req_pkg version path ${req_pkg_name_path} is unobtainable."
     exit 1
  fi

  _COMMON_INC_FILE="${_CURR_PATH}/common_func.inc"
  . "${_COMMON_INC_FILE}"

  check_pkg_ver_deps ${ver_info} ${req_pkg_name} ${req_pkg_name_path}
  ret=$VerCheckStatus

  if [[ "$ret" == "SUCC" ]];then
    log_and_print "[INFO]: ncs with ${req_pkg_name} version relationships check success"
        return 0
  else
    log_and_print "[WARNING]: ncs with ${req_pkg_name} version relationships check failed. \
do you want to continue.  [y/n] "
    while true
    do
      read yn
      if [[ "$yn" == "n" ]]; then
         echo "[INFO]: stop installation!"
         exit 1
      elif [ "$yn" = y ]; then
         break;
      else
         echo "[WARNING]: Input error, please input y or n to choose!"
      fi
    done
  fi
  return 0
}

version_check
exit 0
