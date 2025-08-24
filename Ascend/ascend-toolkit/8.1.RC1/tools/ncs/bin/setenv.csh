#!/usr/bin/env csh
# Perform setenv for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

set PACKAGE=ncs
set CURFILE=`readlink -f ${0}`
set CURPATH=`dirname ${CURFILE}`
set INSTALL_PATH=`realpath ${CURPATH}/../..`

echo "$INSTALL_PATH"
set ncs_path="${INSTALL_DIR}/${PACKAGE}/bin"
if ( -d ${ncs_path} ) then
    set ncs_home=`echo ${PATH}`
    set num=`echo ":${ncs_home}:" | grep ":${ncs_path}:" | wc -l`
    if (${num} == 0) then
        if ("-${ncs_home}" == "-") then
            set PATH=${ncs_path}
        else
            set PATH=${ncs_path}:${ncs_home}
        endif
    endif
endif

set ld="${INSTALL_PATH}/${PACKAGE}/lib64"
if ( -d ${ld} ) then
    set ld_library_path=`echo ${LD_LIBRARY_PATH}`
    set num=`echo ":${ld_library_path}:" | grep ":${ld}:" | wc -l`
    if (${num} == 0) then
        if ("-${ld_library_path}" == "-") then
            set LD_LIBRARY_PATH=${ld}
        else
            set LD_LIBRARY_PATH=${ld}:${ld_library_path}
        endif
    endif
endif
