#!/usr/bin/env csh

set PACKAGE=toolkit

if ( "-${argv}" != "-" ) then
    set CUR_FILE=${argv}
else
    set CUR_FILE=`readlink -f $0`
endif
set CUR_DIR=`dirname ${CUR_FILE}`
set version_dir=`cat "$CUR_DIR/../version.info" | grep "version_dir" | cut -d"=" -f2`
if ( "-$version_dir" == "-" ) then
    set INSTALL_DIR=`realpath ${CUR_DIR}/../..`
else
    set INSTALL_DIR=`realpath ${CUR_DIR}/../../../latest`
endif

set toolchain_path="${INSTALL_DIR}/${PACKAGE}"
if ( -d ${toolchain_path} ) then
    set toolchain_home=`echo ${TOOLCHAIN_HOME}`
    set num=`echo ":${toolchain_home}:" | grep ":${toolchain_path}:" | wc -l`
    if (${num} == 0) then
        if ("-${toolchain_home}" == "-") then
            set TOOLCHAIN_HOME=${toolchain_path}
        else
            set TOOLCHAIN_HOME=${toolchain_path}:${toolchain_home}
        endif
    endif
endif

set lib_path="${INSTALL_DIR}/${PACKAGE}/python/site-packages/"
if (-d ${lib_path}) then
    set python_path=`echo ${PYTHONPATH}`
    set num=`echo ":${python_path}:" | grep ":${lib_path}:" | wc -l`
    if (${num} == 0) then
        if ("-${python_path}" == "-") then
            set PYTHONPATH=${lib_path}
        else
            set PYTHONPATH=${lib_path}:${python_path}
        endif
    endif
endif

set op_tools_path="${INSTALL_DIR}/${PACKAGE}/python/site-packages/bin/"
if (-d ${op_tools_path}) then
    set temp_path=`echo ${PATH}`
    set num=`echo ":${temp_path}:" | grep ":${op_tools_path}:" | wc -l`
    if ( ${num} == 0) then
        if ("-${temp_path}" == "-") then
            set PATH=${op_tools_path}
        else
            set PATH=${op_tools_path}:${temp_path}
        endif
    endif
endif

set msprof_path="${INSTALL_DIR}/${PACKAGE}/tools/profiler/bin/"
if (-d ${msprof_path}) then
    set temp_path=`echo ${PATH}`
    set num=`echo ":${temp_path}:" | grep ":${msprof_path}:" | wc -l`
    if ( ${num} == 0) then
        if ("-${temp_path}" == "-") then
            set PATH=${msprof_path}
        else
            set PATH=${msprof_path}:${temp_path}
        endif
    endif
endif

set ascend_ml_path="${INSTALL_DIR}/${PACKAGE}/tools/aml/lib64"
if (-d ${ascend_ml_path}) then
    set temp_path=`echo ${LD_LIBRARY_PATH}`
    set num=`echo ":${temp_path}:" | grep ":${ascend_ml_path}:" | wc -l`
    if (${num} -eq 0) then
        if ("-${temp_path}" = "-") then
            set LD_LIBRARY_PATH=${ascend_ml_path}
        else
            set LD_LIBRARY_PATH=${ascend_ml_path}:${ascend_ml_path}/plugin:${temp_path}
        fi
    fi
fi

set asys_path="${INSTALL_DIR}/${PACKAGE}/tools/ascend_system_advisor/asys"
if (-d ${asys_path}) then
    set temp_path=`echo ${PATH}`
    set num=`echo ":${temp_path}:" | grep ":${asys_path}:" | wc -l`
    if (${num} -eq 0) then
        if ("-${temp_path}" = "-") then
            set PATH=${asys_path}
        else
            set PATH=${asys_path}:${temp_path}
        endif
    endif
endif

set ccec_compiler_path="${INSTALL_DIR}/${PACKAGE}/tools/ccec_compiler/bin"
if (-d ${ccec_compiler_path}) then
    set temp_path=`echo ${PATH}`
    set num=`echo ":${temp_path}:" | grep ":${ccec_compiler_path}:" | wc -l`
    if (${num} -eq 0) then
        if ("-${temp_path}" = "-") then
            set PATH=${ccec_compiler_path}
        else
            set PATH=${ccec_compiler_path}:${temp_path}
        fi
    fi
fi

set msobjdump_path="${INSTALL_DIR}/${PACKAGE}/tools/msobjdump/"
if (-d ${msobjdump_path}) then
    set temp_path=`echo ${PATH}`
    set num=`echo ":${temp_path}:" | grep ":${msobjdump_path}:" | wc -l`
    if ( ${num} == 0) then
        if ("-${temp_path}" == "-") then
            set PATH=${msobjdump_path}
        else
            set PATH=${msobjdump_path}:${temp_path}
        endif
    endif
endif

set show_kernel_debug_data_tool_path="${INSTALL_DIR}/${PACKAGE}/tools/show_kernel_debug_data/"
if (-d ${show_kernel_debug_data_tool_path}) then
    set temp_path=`echo ${PATH}`
    set num=`echo ":${temp_path}:" | grep ":${show_kernel_debug_data_tool_path}:" | wc -l`
    if ( ${num} == 0) then
        if ("-${temp_path}" == "-") then
            set PATH=${show_kernel_debug_data_tool_path}
        else
            set PATH=${show_kernel_debug_data_tool_path}:${temp_path}
        endif
    endif
endif