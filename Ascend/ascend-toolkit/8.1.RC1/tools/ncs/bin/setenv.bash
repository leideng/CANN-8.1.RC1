#!/usr/bin/env bash
# Perform setenv for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

PACKAGE=ncs
CUR_DIR=`dirname ${BASH_SOURCE[0]}`
INSTALL_DIR=`realpath ${CUR_DIR}/../..`

ncs_path="${INSTALL_DIR}/${PACKAGE}/bin"
if [ -d ${ncs_path} ]; then
    ncs_home=`echo ${PATH}`
    num=`echo ":${ncs_home}:" | grep ":${ncs_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${ncs_home}" = "-" ]; then
            export PATH=${ncs_path}
        else
            export PATH=${ncs_path}:${ncs_home}
        fi
    fi
fi

lib_path="${INSTALL_DIR}/${PACKAGE}/lib64"
if [ -d "${lib_path}" ]; then
    ld_library_path=`echo ${LD_LIBRARY_PATH}`
    num=`echo ":${ld_library_path}:" | grep ":${lib_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${ld_library_path}" = "-" ]; then
            export LD_LIBRARY_PATH=${lib_path}
        else
            export LD_LIBRARY_PATH=${lib_path}:${ld_library_path}
        fi
    fi
fi
