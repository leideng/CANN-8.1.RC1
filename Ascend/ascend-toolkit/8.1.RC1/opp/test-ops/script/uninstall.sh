#!/bin/bash
# Copyright © Huawei Technologies Co., Ltd. 2010-2020. All rights reserved.

install_path=$(realpath "$(dirname $0)/../../..")
latest_path=$(dirname $(realpath "$(dirname $0)/../../.."))/latest
version_path=${install_path}
parent_install_path=${install_path}/opp
dmipath="/opp/test-ops"
install_path=${install_path}${dmipath}
dstscript=${install_path}/script
dstlib64=${install_path}/lib64
arch_dir="$(arch)-linux"

if [ -d "${version_path}/aarch64-linux/lib64" ]&&[ ! -d "${version_path}/x86_64-linux/lib64" ]; then
    arch_dir="aarch64-linux"
elif [ -d "${version_path}/x86_64-linux/lib64" ]&&[ ! -d "${version_path}/aarch64-linux/lib64" ]; then
    arch_dir="x86_64-linux"
fi

__remove_uninstall_package()
{
    if [ -f "${version_path}/cann_uninstall.sh" ]; then
        sed -i "/uninstall_package \"opp\/test-ops\/script\"/d" "${version_path}/cann_uninstall.sh"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove ${version_path}/cann_uninstall.sh uninstall_package command failed!"
            exit 1
        fi
    fi
}

info_log()
{
    echo "[test-ops] [$(date +"%Y-%m-%d %H:%M:%S")] [INFO]: $1"
    return 0
}

if [ -e ${version_path}/${arch_dir}/lib64/train_matmul_model.o ];then
  files=("train_matmul_model.o" "train_matmul_model_fp32.o" "vector_multiplication.o" "vector_multiplication_fp32.o"
  "d2d_bandwidth_test.o" "infer_mmad_model_int8.o" "infer_mmad_model.o" "infer_mmad_v200_model_int8.o" "infer_mmad_v200_model.o"
  "train_mmad_model_int8.o" "train_mmad_model.o" "train_mmad_vec_model_int8.o" "train_mmad_vec_model.o" "train_matmul_model_hf32.o"
  "infer_310b_model.o" "infer_310b_model_int8.o" "d2d_bandwidth_test_310b.o" "train_matmul_model_int8.o" "hbm_test_0.o" "hbm_test_1.o"
  "hbm_test_2.o" "hbm_test_3.o" "hbm_test_4.o" "hbm_test_5.o" "hbm_power.o" "d2d_bandwidth_write_test.o")
fi

for file in ${files[@]}
do  
    if [ "$UID" != "0" ]; then
        if [ ! -w ${version_path}/${arch_dir}/lib64 ]; then
            chmod u+w ${version_path}/${arch_dir}/lib64
            rm -rf ${version_path}/${arch_dir}/lib64/${file}
            chmod u-w ${version_path}/${arch_dir}/lib64
        else
            rm -rf ${version_path}/${arch_dir}/lib64/${file}
        fi
    fi
    if [ "$UID" = "0" ]; then
        rm -rf ${latest_path}/${arch_dir}/lib64/${file}
        rm -rf ${version_path}/${arch_dir}/lib64/${file}
        rm -rf ${install_path}
    fi
done
if [ "$UID" != "0" ]; then
    chmod u+w ${install_path}/script
    chmod u+w ${install_path}/lib64
    chmod u+w ${install_path}
    chmod u+w ${parent_install_path}
    rm -rf ${install_path}
    chmod u-w ${parent_install_path}
fi
chmod -R u+w ${parent_install_path}
if [ -z "$(ls -A ${parent_install_path})" ]; then
    rm -rf ${parent_install_path}
fi
common_arch=${version_path}/${arch_dir}
chmod -R u+w ${common_arch}
if [ -z "$(ls -A ${common_arch}/lib64)" ]; then
    chmod u+w ${common_arch}
    rm -rf ${common_arch}/lib64
    chmod u-w ${common_arch}
fi
if [ -z "$(ls -A ${common_arch})" ]; then
    rm -rf ${common_arch}
fi
 
chmod u+w ${version_path}/cann_uninstall.sh
__remove_uninstall_package
chmod u-w ${version_path}/cann_uninstall.sh
num=$(grep "^uninstall_package " ${version_path}/cann_uninstall.sh | wc -l )
if [ ${num} -eq 0 ]; then
    rm -f "${version_path}/cann_uninstall.sh" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log "WARNING" "Delete file:cann_uninstall.sh failed, please delete it by yourself."
    fi
fi
 
if [ -z "$(ls -A ${version_path})" ]; then
    rm -rf ${version_path}
fi 

opp_path=$(readlink -f $(dirname $(dirname $(dirname ${install_path})))/latest/opp)
CURRENT_VERSION=$(basename $(dirname ${opp_path}))
version_info=${CURRENT_VERSION}
# 依赖runtime维护的version.cfg文件中的runtime_installed_version字段
# CANN同时安装了多个版本, 从runtime_installed_version字段中获取CANN的上一个版本
version_cfg_path=$latest_path/version.cfg
if [ -f "$version_cfg_path" ]; then
    version_line=$(sed -n '/runtime_installed_version/p' $version_cfg_path)
    if [[ $version_line =~ \[(.*)\] ]]; then
        version_info_fisrt=$(echo $version_line | awk -F '[ [ : ]' '{print $NF}' | tr -d ']')
        version_info_second=$(echo $version_line | awk -F '[ [ : ]' '{print $(NF - 2)}' | tr -d ']')
        version_info=${version_info_fisrt}
        # 升级场景，runtime_installed_version字段只有一个版本号，且为当前版本号
        # 所以第二次匹配结果为"runtime_installed_version="
        if [[ ${version_info} = ${CURRENT_VERSION} && ${version_info_second} != "runtime_installed_version=" ]]; then
            version_info=${version_info_second}
        fi
    fi
fi

lib64_arch_path=$(dirname $(dirname ${opp_path}))/latest/${arch_dir}/lib64
if [ -L "${latest_path}/opp" ]&&[ ${version_info} != $(basename ${version_path}) ]; then
    for fl in ${files[@]}
    do
        filename=${fl##*/}
        chmod u+w ${lib64_arch_path}
        ln -sf ../../../${version_info}/${arch_dir}/lib64/${filename} ${lib64_arch_path}/${filename}
        chmod u-w ${lib64_arch_path}
    done
    ln -sf ../${version_info}/opp $(dirname $(dirname ${opp_path}))/latest/
    ln -sf ../${version_info}/opp/test-ops $(dirname $(dirname ${opp_path}))/latest/
else
    for file in ${files[@]}
    do
        if [ "$UID" != "0" ]; then
            if [ ! -w ${latest_path}/${arch_dir}/lib64 ]; then
                chmod u+w ${latest_path}/${arch_dir}/lib64
                rm -rf ${latest_path}/${arch_dir}/lib64/${file}
                chmod u-w ${latest_path}/${arch_dir}/lib64
            else
                rm -rf ${latest_path}/${arch_dir}/lib64/${file}
            fi
        fi
    done
    rm -rf ${latest_path}/test-ops
    rm -rf ${version_path}/test-ops
    if [ -d "${latest_path}/${arch_dir}/lib64" ]; then
        if [ -z "$(ls -A ${latest_path}/${arch_dir}/lib64)" ]; then
            if [ "$UID" != "0" ];then
                if [ ! -w ${latest_path}/${arch_dir} ]; then
                    chmod u+w ${latest_path}/${arch_dir}
                    rm -rf ${latest_path}/${arch_dir}/lib64
                    chmod u-w ${latest_path}/${arch_dir}
                else
                    rm -rf ${latest_path}/${arch_dir}/lib64
                fi
            else
                rm -rf ${latest_path}/${arch_dir}/lib64
            fi
        fi
    else
        info_log "${latest_path}/${arch_dir}/lib64 has been uninstalled in the previous subpackage"
    fi
    
    if [ -d "${latest_path}/${arch_dir}" ]; then
        if [ -z "$(ls -A ${latest_path}/${arch_dir})" ]; then
            rm -rf ${latest_path}/${arch_dir}
        fi
    else
        info_log "${latest_path}/${arch_dir} has been uninstalled in the previous subpackage"
    fi
    
    if [ -d "${latest_path}" ]; then
        if [ -z "$(ls -A ${latest_path})" ]; then
            rm -rf ${latest_path}
        fi
    else
        info_log "${latest_path} has been uninstalled in the previous subpackage"
    fi
fi
info_log "uninstall successfully"