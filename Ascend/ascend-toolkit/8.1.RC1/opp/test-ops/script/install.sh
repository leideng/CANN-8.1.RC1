#!/bin/bash
# Copyright © Huawei Technologies Co., Ltd. 2010-2020. All rights reserved.


srclib64="./lib64/*.o"
srcscript="./script/*.sh"
dmipath="/opp/test-ops"
arch_dir="$(arch)-linux"
install_username=$(id -nu)
install_usergroup=$(id -ng)

if [ "$UID" = "0" ]; then
    install_for_all="y"
fi

function info_log()
{
    echo "[test-ops] [$(date +"%Y-%m-%d %H:%M:%S")] [INFO]: $1"
    return 0
}

function error_log()
{
    echo "[test-ops] [$(date +"%Y-%m-%d %H:%M:%S")] [ERROR]: $1"
    exit 1
}

function create_symbolic()
{
    local lib64_path=${version_path}/${arch_dir}/lib64
    local arch_path=$(dirname $(dirname $(dirname ${install_path})))/latest/${arch_dir}
    local lib64_arch_path=${arch_path}/lib64
    if [ ! -d "${lib64_arch_path}" ]; then
        mkdir -p ${lib64_arch_path}
    fi
    chmod 755 ${arch_path}
    chmod 755 ${lib64_arch_path}
    local latest_path=$(dirname $(dirname $(dirname ${install_path})))/latest
    if [ -L "${latest_path}/test-ops" ]; then
        rm -rf ${latest_path}/test-ops
    fi
    if [ -L ${version_path}/test-ops ]; then
        rm -rf ${version_path}/test-ops
    fi
    ln -sf ../${version_info}/opp/test-ops ${latest_path}/test-ops
    ln -sf ../${version_info}/opp/test-ops ${version_path}/test-ops
    if [ ! -d "${latest_path}" ]; then
        mkdir -p ${latest_path}
    fi
    if [ ! -d "${lib64_path}" ]; then
        mkdir -p ${lib64_path}
    fi

    local dest_arch_dir=${version_path}/${arch_dir}
    local dest_arch_dir_right=$(stat -c %a ${dest_arch_dir})
    chmod 750 ${dest_arch_dir}
    local dest_lib64_dir_right=$(stat -c %a ${lib64_path})
    chmod 750 ${lib64_path}

    local parent_path_right=$(stat -c %a ${parent_install_path})
    chmod 750 ${parent_install_path}
    local test_ops_dir_right=$(stat -c %a ${install_path})
    chmod 750 ${install_path}
    local src_lib64=${install_path}/lib64
    local src_lib64_dir_right=$(stat -c %a ${src_lib64})
    chmod 750 ${src_lib64}
    ln -sf ./${arch_dir}/lib64 ${latest_path}

    
    for fl in $(find ${src_lib64} -name "*.o")
    do
        if [ ! -f ${lib64_path}/$(basename ${fl}) ]; then
            mv -f ${fl} ${lib64_path}
        fi
        filename=${fl##*/}
        ln -sf ../../../../${version_info}/${arch_dir}/lib64/${filename} ${fl}
        ln -sf ../../../${version_info}/${arch_dir}/lib64/${filename} ${lib64_arch_path}/${filename}

    done

    chmod ${dest_lib64_dir_right} ${lib64_path}
    chmod ${dest_arch_dir_right} ${dest_arch_dir}
    chmod ${src_lib64_dir_right} ${src_lib64}
    chmod ${test_ops_dir_right} ${install_path}
    chmod ${parent_path_right} ${parent_install_path}
    if [ "$UID" = "0" ]; then
        chmod 555 ${arch_path}
        chmod 555 ${lib64_arch_path}
    else
        chmod 550 ${arch_path}
        chmod 550 ${lib64_arch_path}
    fi 
}

function __remove_uninstall_package()
{
    if [ -f "${version_path}/cann_uninstall.sh" ]; then
        sed -i "/uninstall_package \"opp\/test-ops\/script\"/d" "${version_path}/cann_uninstall.sh"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove ${version_path}/cann_uninstall.sh uninstall_package command failed!"
            exit 1
        fi
    fi
}

function __add_uninstall_package()
{
    if [ -f "${version_path}/cann_uninstall.sh" ]; then
        if [ `grep -c "opp/test-ops/script" ${version_path}/cann_uninstall.sh` -eq '0' ];then
            if [ `grep -c "opp/script" ${version_path}/cann_uninstall.sh` -ne '0' ]; then
                sed -i "/uninstall_package \"opp\/script\"/i uninstall_package \"opp\/test-ops\/script\"" "${version_path}/cann_uninstall.sh"
            else
                sed -i "/^exit /i uninstall_package \"opp\/test-ops\/script\"" "${version_path}/cann_uninstall.sh"
            fi
        else
            __remove_uninstall_package
            if [ `grep -c "opp/script" ${version_path}/cann_uninstall.sh` -ne '0' ]; then
                sed -i "/uninstall_package \"opp\/script\"/i uninstall_package \"opp\/test-ops\/script\"" "${version_path}/cann_uninstall.sh"
            else
                sed -i "/^exit /i uninstall_package \"opp\/test-ops\/script\"" "${version_path}/cann_uninstall.sh"
            fi
        fi
        return 0
    fi
}

function install()
{
    if [ ! -d "${parent_install_path}" ]; then
        mkdir -p ${parent_install_path}
    fi

    local parent_path_right=$(stat -c %a ${parent_install_path})
    chmod u+w ${parent_install_path}

    if [ ! -d "${install_path}" ]; then
        mkdir -p ${install_path}
    fi

    mkdir -p ${dstscript}
    cp ${srcscript} ${dstscript}

    mkdir -p ${dstlib64}
    cp ${srclib64} ${dstlib64}
    version_file_path="${install_path}/version.info"
    scene_file_path="${install_path}/scene.info"
    echo "Version=$(basename ${version_path})" > ${version_file_path}
    echo "os=linux" > ${scene_file_path}
    echo "arch=$(arch)" >> ${scene_file_path}

    if [ -d "${install_path}" ]; then
        file_mode=440
        executable_mode=550

        if [ "$(whoami)" == "root" ]; then
            chown -R ${install_username}:${install_usergroup} ${dstlib64}
        fi

        chmod -R ${executable_mode} ${install_path}
        chmod ${file_mode} ${dstlib64}/*.o

        if [ "$(whoami)" == "root" ]; then
            chmod 555 ${install_path}
        fi
        if [ ! -f "${version_path}/cann_uninstall.sh" ]; then
            cp ${install_path}/script/cann_uninstall.sh ${version_path}
        fi
        chmod 500 ${version_path}/cann_uninstall.sh
        chmod u+w ${version_path}/cann_uninstall.sh
        __add_uninstall_package
        chmod u-w ${version_path}/cann_uninstall.sh
        info_log "your install path is ${install_path}"
        info_log "install is success" 
    else
        info_log "install is failed"
    fi
    if [ "${install_for_all}" == "y" ]; then
        chmod o+r ${install_path}
        chmod o+x ${install_path}
        chmod o+r ${dstlib64}
        chmod o+x ${dstlib64}
        chmod o+r ${dstlib64}/*.o
    fi

    chmod ${parent_path_right} ${parent_install_path}
    if [ -f "${parent_install_path}"/test-ops/script/install.sh ];then
        chmod 500 ${parent_install_path}/test-ops/script/install.sh
    fi
    if [ -f "${parent_install_path}"/test-ops/script/uninstall.sh ];then
        chmod 500 ${parent_install_path}/test-ops/script/uninstall.sh
    fi
    chmod 440 $version_file_path 2> /dev/null
    chmod 440 $scene_file_path 2> /dev/null
    create_symbolic
}

function reinstall() {
    chmod -R 750 ${install_path}
    for file in ${ops_files[@]}
    do
        if [ ! -w ${version_path}/${arch_dir}/lib64 ]; then
             chmod u+w ${version_path}/${arch_dir}/lib64/${file}
        fi
    done
    uninstall
    install
    for file in ${ops_files[@]}
    do
        chmod u-w ${version_path}/${arch_dir}/lib64
    done
    info_log "re-install successfully"
}

function remove_invalid_symbolic_link() {
    # 升级过程中，调用8.0.0版本以前的uninstall.sh脚本执行，会产生无效软链接，需额外进行删除。该函数可在9.0.0版本进行删除
    local opp_invalid_symbolic_link=$(dirname $(dirname $(dirname ${install_path})))/latest/opp/opp
    if [ -L "${opp_invalid_symbolic_link}" ]; then
       # 卸载latest/opp/opp无效软链接
       rm -rf "${opp_invalid_symbolic_link}"
    fi
}

function chmod_version_opp_path() {
    # 卸载过程中，调用8.0.0版本以前的uninstall.sh脚本，会在opp目录下再创建一个opp无效软链接
    # 非root用户下，opp目录权限为550，会创建失败，因此临时修改opp目录权限。该函数可在9.0.0版本进行删除
    local change_permission=$1
    local latest_opp_symbolic_link=$(dirname $(dirname $(dirname ${install_path})))/latest/opp
    if [ -L "${latest_opp_symbolic_link}" ]; then
        local version_opp_path=$(readlink -f "${latest_opp_symbolic_link}")
        if [[ -z "${change_permission}" ]]; then
          # 若未指定权限，将opp目录还原成原始权限
          chmod ${version_opp_permission} $version_opp_path
        else
          # 记录opp目录原始权限，并修改为指定权限
          version_opp_permission=$(stat -c "%a %n" "$version_opp_path")
          chmod ${change_permission} $version_opp_path
        fi
    fi
}

function uninstall() {
  chmod_version_opp_path 750
    # 使用已安装软件包路径下的uninstall.sh进行卸载
    latest_test_ops_path=$(dirname $(dirname $(dirname ${install_path})))/latest/test-ops
    ${latest_test_ops_path}/script/uninstall.sh
    remove_invalid_symbolic_link
    chmod_version_opp_path
    info_log "uninstall is success"
}

function upgrade() {
    if [ ! -d "${last_test_ops_path}" ]; then
         error_log "Update failed, Incorrect address or incomplete command,The <--upgrade> command needs to be used with the <--install-path=> command to specify the updated tool path"
         return
    fi
    uninstall
    install
    info_log "upgrade successfully"
}

function check_install_status() {
    if [ "${is_upgrade}" == "y" ] || [ "${is_install}" == "y" ]; then
      local install_path=$1
      local install_info_path="${install_path}"/version.info
      local installed_info_num
      installed_info_num=$(ls "${install_info_path}" 2>/dev/null | wc -l)
      # 路径中已安装同架构的其他版本，升级安装
      if [ "${installed_info_num}" -gt 0 ]&&[[ -z $(find $(dirname ${version_path}) -name ${version_info}) ]]; then
          is_install=n
          is_upgrade=y
      # 路径中已安装同架构的同版本，重新覆盖安装
      elif [ "${installed_info_num}" -gt 0 ]&&[[ -n $(find $(dirname ${version_path}) -name ${version_info}) ]]; then
          re_install=y
      # 路径中未安装同架构任何版本，直接安装
      elif [ "${installed_info_num}" -eq 0 ]; then
          return
      fi
    fi
}

# 解析脚本输入参数
function parse_script_args() {
  while true
  do
      case "$3" in
          --check)
             exit 0
             ;;
          --upgrade)
              is_upgrade=y
              re_install=n
              shift
              ;;
          --uninstall)
              is_uninstall=y
              shift
              ;;
          --install-path=*)
              is_install_path=y
              echo ${3}
              check_null=${3}
              install_path=$(echo $3 | cut -d"=" -f2 )
              version_info=$(cat ./version_info)
              info_log "new_version_info ${version_info}"
              install_path=${install_path}/${version_info}
              version_path=${install_path}
              parent_install_path=${install_path}/opp
              install_path=${install_path}${dmipath}
              dstscript=${install_path}/script
              dstlib64=${install_path}/lib64
              shift
              ;;
          --install)
              is_install=y
              re_install=n
              shift
              ;;
          --run)
              is_install=y
              shift
              ;;
          --quiet)
              shift
              ;;
          --full)
              is_install=y
              shift
              ;;
          --devel)
              is_install=y
              shift
              ;;
          --install-for-all)
              install_for_all=y
              shift
              ;;
              *)
              break
              ;;
          esac
  done
}

# 脚本入参的相关处理函数
function check_script_args() {
  # 安装为相对路径时报错: 1） 包含 ../或./ 2) 目录名不为/开头
  if [[ "${install_path}" =~ /\/\./ || "${install_path}" =~ /\/\.\./ || "${install_path:0:1}" != "/" ]]; then
      is_install_path=n
      error_log "Please follow the installation address after the --install-path=<Absolute path>"
  fi

  # 指定--install参数但未指定安装路径--install-path参数
  if [ "${is_install}" == "y" ]&&[ x"${is_install_path}" == "x" ]; then
      error_log "Only the <--install> or <--full> command can't tell me you hit the installation directory Please enter the <--install-path=> command to tell me the directory where you want to install"
  fi

  # 指定--upgrade参数但未指定安装路径--install-path参数
  if [ "${is_upgrade}" == "y" ]&&[ "x""${check_null}" == "x" ]; then
      error_log "The path of the update tool is empty or the command input is incorrect,\
      The <--upgrade> command needs to be used with the <--install-path=> command to specify the updated tool path"
  fi

  # 指定--uninstall参数但未指定安装路径--install-path参数
  if [ "${is_uninstall}" == "y" ]&&[ "x""${check_null}" == "x" ]; then
      error_log "The path of the install tool is empty or the command input is incorrect,\
      The <--uninstall> command needs to be used with the <--install-path=> command to specify the tool path"
  fi
}

# 补齐具体执行安装，升级，卸载等流程需要的参数信息
function complete_params() {
  last_test_ops_path=$(readlink -f $(dirname $(dirname $(dirname ${install_path})))/latest/test-ops)
  last_parent_install_path=$(dirname ${last_test_ops_path})
  last_version_path=$(dirname ${last_parent_install_path})

  # 根据已安装包信息，获取架构信息
  if [ -d "${last_version_path}/aarch64-linux/lib64" ]&&[ ! -d "${last_version_path}/x86_64-linux/lib64" ]; then
      arch_dir="aarch64-linux"
  elif [ -d "${last_version_path}/x86_64-linux/lib64" ]&&[ ! -d "${last_version_path}/aarch64-linux/lib64" ]; then
      arch_dir="x86_64-linux"
  fi

  # 检查已安装状态，调整参数
  check_install_status "${install_path}"

  # 算子文件
  if [ -e ${version_path}/${arch_dir}/lib64/train_matmul_model.o ];then
    ops_files=("train_matmul_model.o" "train_matmul_model_fp32.o" "vector_multiplication.o" "vector_multiplication_fp32.o"
    "d2d_bandwidth_test.o" "infer_mmad_model_int8.o" "infer_mmad_model.o" "infer_mmad_v200_model_int8.o" "infer_mmad_v200_model.o"
    "train_mmad_model_int8.o" "train_mmad_model.o" "train_mmad_vec_model_int8.o" "train_mmad_vec_model.o" "train_matmul_model_hf32.o"
    "infer_310b_model.o" "infer_310b_model_int8.o" "d2d_bandwidth_test_310b.o" "train_matmul_model_int8.o" "hbm_test_0.o" "hbm_test_1.o"
    "hbm_test_2.o" "hbm_test_3.o" "hbm_test_4.o" "hbm_test_5.o" "hbm_power.o" "d2d_bandwidth_write_test.o")
  fi
}

function process() {
    # reinstall安装
    if [ "${is_install_path}" == "y" ]&&[ "${re_install}" == "y" ]; then
        reinstall
        exit 0
    fi

    # install安装
    if [ "${is_install_path}" == "y" ]&&[ "${is_install}" == "y" ]; then
        install
        exit 0
    fi

    # uninstall卸载
    if [ "${is_uninstall}" == "y" ]&&[ "x""${check_null}" != "x" ]; then
        uninstall
        exit 0
    fi

    # upgrade更新
    if [ "${is_upgrade}" == "y" ]&&[ "${is_install_path}" == "y" ]&&[ "${re_install}" == "n" ]; then
        upgrade
        exit 0
    fi
}

parse_script_args $*
check_script_args $*
complete_params
process