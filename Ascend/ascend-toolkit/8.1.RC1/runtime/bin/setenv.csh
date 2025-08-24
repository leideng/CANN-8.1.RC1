#!/usr/bin/env csh
# Perform setenv for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

set CURFILE=`readlink -f ${1}`
set CURPATH=`dirname ${CURFILE}`
set DEP_HAL_NAME="libascend_hal.so"
set DEP_INFO_FILE="/etc/ascend_install.info"
set IS_INSTALL_DRIVER="n"

set install_info="$CURPATH/../ascend_install.info"
set hetero_arch=`grep -i "Runtime_Hetero_Arch_Flag=" "$install_info" | cut --only-delimited -d"=" -f2`
if ( "$2" == "multi_version" ) then
    if ( "$hetero_arch" == "y" ) then
        set INSTALL_DIR="`realpath ${CURPATH}/../../../../../latest`/runtime"
    else
        set INSTALL_DIR="`realpath ${CURPATH}/../../../latest`/runtime"
    endif
else
    set INSTALL_DIR="`realpath ${CURPATH}/..`"
endif

set lib_stub_path="${INSTALL_DIR}/lib64/stub"
set lib_path="${INSTALL_DIR}/lib64"

if ( $?LD_LIBRARY_PATH == 1 ) then
    set ld_library_path_array=`echo $LD_LIBRARY_PATH | tr ":" " "`
    foreach var ($ld_library_path_array)
        if ( -d "$var" ) then
            if ( "$var" =~ *driver* ) then
                if {( find $var | grep -q ${DEP_HAL_NAME} )} then
                    set IS_INSTALL_DRIVER="y"
                endif
            endif
        endif
    end
endif

# 第一种方案判断驱动包是否存在
if ( -f "$DEP_INFO_FILE" ) then
    set driver_install_path_param=`grep -iw driver_install_path_param "$DEP_INFO_FILE" | cut --only-delimited -d"=" -f2-`
    if ( ! -z "$driver_install_path_param" ) then
        set DEP_PKG_VER_FILE="$driver_install_path_param/driver"
        if ( -d "$DEP_PKG_VER_FILE" ) then
            set DEP_HAL_PATH=`(find "$DEP_PKG_VER_FILE" -name "$DEP_HAL_NAME" > /dev/tty) >& /dev/null`
            if ( ! -z "$DEP_HAL_PATH" ) then
                set IS_INSTALL_DRIVER="y"
            endif
        endif
    endif
endif

# 第二种方案判断驱动包是否存在
echo ":${PATH}:" | grep -q ":/sbin:"
if ( $status != 0 ) then
    setenv PATH "${PATH}:/sbin"
endif

which ldconfig >& /dev/null
if ( $status == 0 ) then
    ldconfig -p | grep "${DEP_HAL_NAME}" >& /dev/null
    if ( $status == 0 ) then
        set IS_INSTALL_DRIVER="y"
    endif
endif

if ( "$IS_INSTALL_DRIVER" == "n" ) then
    if ( -d "${lib_path}" ) then
        set ld_library_path=""
        if ( $?LD_LIBRARY_PATH == 1 ) then
            set ld_library_path="$LD_LIBRARY_PATH"
        endif
        set num=`echo ":${ld_library_path}:" | grep ":${lib_stub_path}:" | wc -l`
        if ( "$num" == 0 ) then
            if ( "-${ld_library_path}" == "-" ) then
                setenv LD_LIBRARY_PATH "${lib_stub_path}"
            else
                setenv LD_LIBRARY_PATH "${ld_library_path}:${lib_stub_path}"
            endif
        endif
    endif
endif

if ( -d "${lib_path}" ) then
    set ld_library_path=""
    if ( $?LD_LIBRARY_PATH == 1 ) then
        set ld_library_path="$LD_LIBRARY_PATH"
    endif
    set num=`echo ":${ld_library_path}:" | grep ":${lib_path}:" | wc -l`
    if ( "$num" == 0 ) then
        if ( "-${ld_library_path}" == "-" ) then
            setenv LD_LIBRARY_PATH "${lib_path}"
        else
            setenv LD_LIBRARY_PATH "${lib_path}:${ld_library_path}"
        endif
    endif
endif

set lib_path="`realpath ${INSTALL_DIR}/..`"
set lib_path="${lib_path}/pyACL/python/site-packages/acl"
if ( -d "${lib_path}" ) then
    set python_path=""
    if ( $?PYTHONPATH == 1 ) then
        set python_path="$PYTHONPATH"
    endif
    set num=`echo ":${python_path}:" | grep ":${lib_path}:" | wc -l`
    if ( "$num" == 0 ) then
        if ( "-${python_path}" == "-" ) then
            setenv PYTHONPATH "${lib_path}"
        else
            setenv PYTHONPATH "${lib_path}:${python_path}"
        endif
    endif
endif

set custom_path_file="$INSTALL_DIR/../conf/path.cfg"
set common_interface="$INSTALL_DIR/script/common_interface.csh"
set owner="`stat -c %U $CURFILE`"
if ( "`id -u`" != 0 && "`id -un`" != "$owner" && -f "$custom_path_file" && -f "$common_interface" ) then
    csh -f "$common_interface" mk_custom_path "$custom_path_file"
endif
