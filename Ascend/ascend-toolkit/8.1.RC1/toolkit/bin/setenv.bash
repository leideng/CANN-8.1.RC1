#!/bin/bash
PACKAGE=toolkit
CUR_DIR=`dirname ${BASH_SOURCE[0]}`

version_dir=`cat "$CUR_DIR/../version.info" | grep "version_dir" | cut -d"=" -f2`
if [ -z "$version_dir" ]; then
    INSTALL_DIR=`realpath ${CUR_DIR}/../..`
else
    INSTALL_DIR=`realpath ${CUR_DIR}/../../../latest`
fi

toolchain_path="${INSTALL_DIR}/${PACKAGE}"
if [ -d ${toolchain_path} ]; then
    toolchain_home=`echo ${TOOLCHAIN_HOME}`
    num=`echo ":${toolchain_home}:" | grep ":${toolchain_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${toolchain_home}" = "-" ]; then
            export TOOLCHAIN_HOME=${toolchain_path}
        else
            export TOOLCHAIN_HOME=${toolchain_path}:${toolchain_home}
        fi
    fi
fi

lib_path="${INSTALL_DIR}/${PACKAGE}/python/site-packages/"
if [ -d ${lib_path} ]; then
    python_path=`echo ${PYTHONPATH}`
    num=`echo ":${python_path}:" | grep ":${lib_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${python_path}" = "-" ]; then
            export PYTHONPATH=${lib_path}
        else
            export PYTHONPATH=${lib_path}:${python_path}
        fi
    fi
fi

op_tools_path="${INSTALL_DIR}/${PACKAGE}/python/site-packages/bin/"
if [ -d ${op_tools_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${op_tools_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${op_tools_path}
        else
            export PATH=${op_tools_path}:${temp_path}
        fi
    fi
fi

msprof_path="${INSTALL_DIR}/${PACKAGE}/tools/profiler/bin/"
if [ -d ${msprof_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${msprof_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${msprof_path}
        else
            export PATH=${msprof_path}:${temp_path}
        fi
    fi
fi

ascend_ml_library_path="${INSTALL_DIR}/${PACKAGE}/tools/aml/lib64"
if [ -d ${ascend_ml_library_path} ]; then
    temp_path=`echo ${LD_LIBRARY_PATH}`
    num=`echo ":${temp_path}:" | grep ":${ascend_ml_library_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export LD_LIBRARY_PATH=${ascend_ml_library_path}
        else
            export LD_LIBRARY_PATH=${ascend_ml_library_path}:${ascend_ml_library_path}/plugin:${temp_path}
        fi
    fi
fi

asys_path="${INSTALL_DIR}/${PACKAGE}/tools/ascend_system_advisor/asys"
if [ -d ${asys_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${asys_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${asys_path}
        else
            export PATH=${asys_path}:${temp_path}
        fi
    fi
fi

ccec_compiler_path="${INSTALL_DIR}/${PACKAGE}/tools/ccec_compiler/bin"
if [ -d ${ccec_compiler_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${ccec_compiler_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${ccec_compiler_path}
        else
            export PATH=${ccec_compiler_path}:${temp_path}
        fi
    fi
fi

ascendc_compiler_path="${INSTALL_DIR}/${PACKAGE}/tools/tikcpp/ascendc_compiler"
if [ -d ${ascendc_compiler_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${ascendc_compiler_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${ascendc_compiler_path}
        else
            export PATH=${ascendc_compiler_path}:${temp_path}
        fi
    fi
fi

msobjdump_path="${INSTALL_DIR}/${PACKAGE}/tools/msobjdump/"
if [ -d ${msobjdump_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${msobjdump_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${msobjdump_path}
        else
            export PATH=${msobjdump_path}:${temp_path}
        fi
    fi
fi

show_kernel_debug_data_tool_path="${INSTALL_DIR}/${PACKAGE}/tools/show_kernel_debug_data/"
if [ -d ${show_kernel_debug_data_tool_path} ]; then
    temp_path=`echo ${PATH}`
    num=`echo ":${temp_path}:" | grep ":${show_kernel_debug_data_tool_path}:" | wc -l`
    if [ ${num} -eq 0 ]; then
        if [ "-${temp_path}" = "-" ]; then
            export PATH=${show_kernel_debug_data_tool_path}
        else
            export PATH=${show_kernel_debug_data_tool_path}:${temp_path}
        fi
    fi
fi