#!/bin/bash
param_mult_ver=$1
REAL_SHELL_PATH=`realpath ${BASH_SOURCE[0]}`
CANN_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../ && pwd)
if [ -d "${CANN_PATH}/opp" ] && [ -d "${CANN_PATH}/../latest" ]; then
    INSATLL_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../../ && pwd)
    if [ -L "${INSATLL_PATH}/latest/opp" ]; then
        _ASCEND_OPP_PATH=`cd ${CANN_PATH}/opp && pwd`
        if [ "$param_mult_ver" = "multi_version" ]; then
            _ASCEND_OPP_PATH=`cd ${INSATLL_PATH}/latest/opp && pwd`
        fi
    elif [ ! -L "${INSATLL_PATH}/latest/opp" ] && [ -L "${INSATLL_PATH}/latest/opp_kernel" ]; then
        _ASCEND_OPP_PATH=`cd ${CANN_PATH}/opp && pwd`
    fi
elif [ -d "${CANN_PATH}/opp" ]; then
    _ASCEND_OPP_PATH=`cd ${CANN_PATH}/opp && pwd`
fi  

_ASCEND_AICPU_PATH=`cd ${_ASCEND_OPP_PATH}/../ && pwd`
export ASCEND_OPP_PATH=${_ASCEND_OPP_PATH}
export ASCEND_AICPU_PATH=${_ASCEND_AICPU_PATH}
export PYTHONPATH=$PYTHONPATH:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/
curfile="$(realpath ${BASH_SOURCE:-$0})"
USER="$(id -un)"
owner=$(stat -c %U "$curfile")
custom_path_file="$_ASCEND_OPP_PATH/../conf/path.cfg"
common_interface="$_ASCEND_OPP_PATH/script/common_interface.bash"
dst_dir="$(grep -w "data" "$custom_path_file" | cut --only-delimited -d"=" -f2-)"
eval "dst_dir=$dst_dir"
vendors_dir="$_ASCEND_OPP_PATH/vendors"
vendor_dir=$(ls "$vendors_dir" 2> /dev/null)
if [ $(id -u) -ne 0 ] && [ "$owner" != "$(whoami)" ] && [ -f "$custom_path_file" ] && [ -f "$common_interface" ]; then
    . "$common_interface"
    mk_custom_path "$custom_path_file"
    for dir_name in "data"; do
        if [ -d "$_ASCEND_OPP_PATH/built-in/$dir_name" ] && [ -d "$dst_dir" ]; then
            chmod -R u+w $dst_dir/* > /dev/null 2>&1
            cp -rfL $_ASCEND_OPP_PATH/built-in/$dir_name/* "$dst_dir"
        fi
        if [ -d "$vendors_dir/$vendor_dir/$dir_name" ] && [ -d "$dst_dir" ]; then
            chmod -R u+w $dst_dir/* > /dev/null 2>&1
            cp -rfL $vendors_dir/$vendor_dir/$dir_name/* "$dst_dir"
        elif [ ! -d "$vendors_dir/$vendor_dir/$dir_name" ] && [ -d "$dst_dir" ] && [ -d "$_ASCEND_OPP_PATH/$dir_name/custom" ]; then
            chmod -R u+w $dst_dir/* > /dev/null 2>&1
            cp -rfL $_ASCEND_OPP_PATH/$dir_name/* "$dst_dir"
        fi
    done
fi
if [ $(id -u) -ne 0 ] && [ "$owner" != "$(whoami)" ]; then
    opp_custom_list="op_impl op_proto fusion_pass fusion_rules framework"
    for i in $opp_custom_list; do
        dst_file=$dst_dir$i/custom
        mkdir -p "$dst_file"
        chmod -R u+w $dst_file > /dev/null 2>&1
        custom_file=$(find $_ASCEND_OPP_PATH/ -name "custom" |grep $i)
        if [ "$custom_file" != "" ] && [ ! -d "$vendors_dir/$vendor_dir/$i" ]; then
            opp_custom_file=$(ls "$custom_file" 2> /dev/null)
        elif [ -d "$vendors_dir/vendor_dir/$i" ];then
            opp_custom_file=$(ls "$vendors_dir/$vendor_dir/$i" 2> /dev/null)
        fi
        if [ "$opp_custom_file" != "" ];then
            cp -rfL $custom_file/*  $dst_file
        else
            echo "[INFO]: $custom_file/ is empty"
        fi
    done
fi

library_path="${_ASCEND_OPP_PATH}/lib64"
kernel_library_path="${INSATLL_PATH}/latest/opp_latest/lib64"

if [ -d "${kernel_library_path}" ]; then
    ld_library_path="${LD_LIBRARY_PATH}"
    has_opp=$(echo ":${ld_library_path}:" | grep ":${library_path}:" | wc -l)
    if [ "${has_opp}" -ne 0 ]; then
        new_ld_library_path=$(echo ${ld_library_path} | sed "s|${library_path}||g" | sed 's|::|:|g' | sed 's|^:||' | sed 's|:$||')
        ld_library_path="${new_ld_library_path}"
    fi
    
    has_opp_latest=$(echo ":${ld_library_path}:" | grep ":${kernel_library_path}:" | wc -l)
    if [ "${has_opp_latest}" -eq 0 ]; then
        if [ "-${ld_library_path}" = "-" ]; then
            export LD_LIBRARY_PATH="${kernel_library_path}"
        else
            export LD_LIBRARY_PATH="${kernel_library_path}:${ld_library_path}"
        fi       
    else
        export LD_LIBRARY_PATH="${ld_library_path}"
    fi

elif [ -d "${library_path}" ]; then
    ld_library_path="${LD_LIBRARY_PATH}"
    num=$(echo ":${ld_library_path}:" | grep ":${library_path}:" | wc -l)
    if [ "${num}" -eq 0 ]; then
        if [ "-${ld_library_path}" = "-" ]; then
            export LD_LIBRARY_PATH="${library_path}"
        else
            export LD_LIBRARY_PATH="${library_path}:${ld_library_path}"
        fi
    fi
fi
