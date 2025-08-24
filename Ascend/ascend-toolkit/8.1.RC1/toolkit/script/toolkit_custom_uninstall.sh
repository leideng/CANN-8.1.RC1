#!/bin/bash
SHELL_DIR=$(cd "$(dirname "$0")" || exit;pwd)
COMMON_SHELL_PATH="$SHELL_DIR/common.sh"
LOG_PATH="/var/log/ascend_seclog/ascend_install.log"
PACKAGE=toolkit
LEVEL_INFO="INFO"
LEVEL_WARN="WARNING"
LEVEL_ERROR="ERROR"

source "${COMMON_SHELL_PATH}"

log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Toolkit] [${cur_date}] [$1]: $content" >> "${log_file}"
}

log_and_print() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Toolkit] [${cur_date}] [$1]: $content"
    echo "[Toolkit] [${cur_date}] [$1]: $content" >> "${log_file}"
}

removeDir() {
    local path_=$1

    if [ ! -d ${path_} ]; then
        log ${LEVEL_INFO} "${path_} has been removed."
        return 0
    fi
    sub_num=`ls -l ${path_} | wc -l`
    if [ ${sub_num} -gt 1 ]; then
        log ${LEVEL_INFO} "The dir ${path_} not empty, can't remove."
        return 0
    fi
    chmod +w ${path_} >/dev/null 2>&1
    rm -rf ${path_} >/dev/null 2>&1
    if [ $? -ne 0 ] || [ -d ${path_} ]; then
        log_and_print ${LEVEL_ERROR} "Remove ${path_} failed."
        return 1
    else
        log ${LEVEL_INFO} "Remove ${path_} succeed."
    fi
    return 0
}

### remove python dir
removePythonLocalDir() {
    if [ ! -d "${install_path}/python" ]; then
        return 0
    fi

    removeDir ${install_path}/"python/site-packages"
    if [ $? -ne 0 ]; then
        return 1
    fi
    if [ -d ${install_path}/"python/site-packages" ]; then
        return 0
    fi
    removeDir ${install_path}/"python"
    if [ $? -ne 0 ]; then
        return 1
    fi
    return 0
}

### remove bin files in python
removePythonLocalBin() {
    local python_path_="$1"
    local module_arr_=`echo $@ | cut -d" " -f2-`

    if [ ! -d "${python_path_}/bin" ]; then
        return 0
    fi
    for item in ${module_arr_[@]}; do
        local file_="${python_path_}/bin/${item}"
        if [ ! -f ${file_} ]; then
            log ${LEVEL_INFO} "${file_} does not exist."
            continue
        fi
        rm -rf ${file_}
        if [ $? -ne 0 ]; then
            log_and_print ${LEVEL_ERROR} "Remove ${item} failed."
            return 1
        fi
        log ${LEVEL_INFO} "Remove ${item} succeed."
        local num_=`ls -l "${python_path_}/bin" | wc -l`
        if [ ${num_} -eq 1 ]; then
            rm -rf "${python_path_}/bin"
            if [ $? -ne 0 ]; then
                log_and_print ${LEVEL_ERROR} "Remove ${python_path_}/bin failed."
                return 1
            fi
            log ${LEVEL_INFO} "Remove ${python_path_}/bin succeed."
        fi
    done
    return 0
}

### uninstall whl
whlUninstallPackage() {
    local module_="$1"
    local python_path_="$2"
    # Root directory name generated after the whl package is installed in the specified path.
    # If parameter 3 is not transferred, the root directory name is the module name by default.
    local module_root_name="$3"
    if [ ! -n "${module_root_name}" ]; then
        module_root_name=${module_}
    fi

    log ${LEVEL_INFO} "start to uninstall ${module_}"
    log ${LEVEL_INFO} "The path ${python_path_} of ${module_root_name} whl package to be uninstalled"
    if [ -d "${python_path_}/${module_root_name}" ]; then
        export PYTHONPATH=${python_path_}
    else
        unset PYTHONPATH
    fi
    pip3 uninstall -y "${module_}" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "uninstall ${module_} failed."
        return 1
    fi
    log ${LEVEL_INFO} "uninstall ${module_} succeed."
    return 0
}

uninstallOpPython() {
    local module_arr_=(op_ut_run op_ut_helper msopst msopst.ini)
    removePythonLocalBin ${install_path}/python/site-packages ${module_arr_[@]}
    [ $? -ne 0 ] && return 1

    # remove op generator
    whlUninstallPackage op_gen ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1

    module_arr_=(msopgen)
    removePythonLocalBin ${install_path}/python/site-packages ${module_arr_[@]}
    [ $? -ne 0 ] && return 1

    whlUninstallPackage msobjdump ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1

    whlUninstallPackage show_kernel_debug_data ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1
    return 0
}

uninstallAllPython() {
    checkAllFeature ${feature_type}
    [ $? -ne 0 ] && return 0

        whlUninstallPackage op_test_frame ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1

    local module_arr_=(op_ut_run op_ut_helper msopst msopst.ini)
    removePythonLocalBin ${install_path}/python/site-packages ${module_arr_[@]}
    [ $? -ne 0 ] && return 1

    # remove op generator
    whlUninstallPackage op_gen ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1

    module_arr_=(msopgen)
    removePythonLocalBin ${install_path}/python/site-packages ${module_arr_[@]}
    [ $? -ne 0 ] && return 1

    # remove hccl parser
    whlUninstallPackage hccl_parser ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1

    whlUninstallPackage msobjdump ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1

    whlUninstallPackage show_kernel_debug_data ${install_path}/python/site-packages
    [ $? -ne 0 ] && return 1
    return 0
}

uninstallMsprofPython() {
    # remove msprof
    changeDirMode 750 ${install_path}/tools/profiler/profiler_tool
    changeFileMode 750 ${install_path}/tools/profiler/profiler_tool
    whlUninstallPackage msprof ${install_path}/tools/profiler/profiler_tool analysis
    [ $? -ne 0 ] && return 1
    return 0
}

uninstallPython() {
    local _py_path="$install_path/$PACKAGE/python"
    # remove python softlink in toolkit
    if [ -d "$_py_path" ]; then
        rm -rf "$_py_path"
    fi

    uninstallOpPython
    [ $? -ne 0 ] && return 1

    uninstallAllPython
    [ $? -ne 0 ] && return 1

    uninstallMsprofPython
    [ $? -ne 0 ] && return 1

    # remove python dir
    removePythonLocalDir
    [ $? -ne 0 ] && return 1
    return 0
}

init() {
    [ ! -d "${install_path}" ] && exit 1

    if [ ! -z "${version_dir}" ]; then
        install_path="${install_path}/${version_dir}"
        [ ! -d "${install_path}" ] && exit 1
    fi

    if [ $(id -u) -eq 0 ]; then
        log_file=${LOG_PATH}
    else
        local _home_path=$(eval echo "~")
        log_file="${_home_path}/${LOG_PATH}"
    fi
}

log_file=""
is_quiet=n
pylocal=n
install_path=""
version_dir=""

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        [ -z "${install_path}" ] && exit 1
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2-)
        [ -z "${version_dir}" ] && exit 1
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --pylocal=*)
        pylocal=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --feature=*)
        feature_type=$(echo "$1" | cut -d"=" -f2)
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

init

uninstallPython
[ $? -ne 0 ] && exit 1

exit 0
