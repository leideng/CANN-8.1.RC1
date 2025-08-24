#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------


import os
import sys
import argparse
import shutil
import subprocess

from common import Logger as logger2
from common import get_opp_path
from common import verify_pathfile_exist_permission
from common import rmtree_file

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True

SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR_PATH = os.path.realpath(os.path.join(SCRIPT_DIR_PATH, "..", ".."))
CURRENT_PATH = os.getcwd()


def _args_parse():
    """
    :return: 外界传入的参数值
    """
    parser = argparse.ArgumentParser(description='Parses the input parameters')
    parser.add_argument('-i', '--install', metavar='install', dest='install', nargs='?', const='',
                        default='', help='install debug.run')
    return parser.parse_args(sys.argv[3:])


def _check_debug_kernel_exist(install_path):
    """
    :param install_path: 安装路径
    :return: True/False
    """
    debug_kernel_dir = os.path.join(install_path, "debug_kernel")
    debug_kernel_dir_state = verify_pathfile_exist_permission(debug_kernel_dir, "path", "erwx")
    if not debug_kernel_dir_state.exist_state:
        logger2.info(f"[{debug_kernel_dir}] path does not exist.")
        return False
    logger2.info(f"[{debug_kernel_dir}] path does exist.")
    return True


def install_process() -> bool:
    """
    :return: True/False
    """
    # 设置日志格式，安装参数
    logger2.setting_log()
    logger2.info(f"start installing ...")
    _args_parse()
    flag, ascend_opp_path = get_opp_path()
    if not flag:
        return False
    install_path = os.path.realpath(ascend_opp_path)
    logger2.info(f"target installation path is {install_path}.")

    # 判断首次安装还是覆盖安装(先卸载)
    if _check_debug_kernel_exist(install_path):
        if verify_pathfile_exist_permission(os.path.join(install_path, "debug_kernel", "uninstall.sh"), \
            "file", "erwx").exist_state:
            uninstall_cmd = f"bash {os.path.join(install_path, 'debug_kernel', 'uninstall.sh')}"
            result = subprocess.run(uninstall_cmd, shell=True, check=False)
            if result.returncode != 0:
                logger2.warning("failed to uninstall debug_kernel.")
                if not rmtree_file(os.path.join(install_path, "debug_kernel")):
                    return False
        else:
            if not rmtree_file(os.path.join(install_path, "debug_kernel")):
                return False
        logger2.info(f"the existing debug_kernel installation package is uninstalled successfully.")
    
    # 复制文件
    try:
        shutil.copytree(os.path.join(PACKAGE_DIR_PATH, "debug_kernel"), os.path.join(install_path, "debug_kernel"))
        logger2.info(f"debug_kernel files is successfully copied to the \
            {os.path.join(install_path, 'debug_kernel')} path.")
    except Exception as e:
        logger2.exception(f"failed to copy debug_kernel files to [{os.path.join(install_path, 'debug_kernel')}], {e}")
        return False
    
    logger2.info(f"installation complete.")
    return True
    

if __name__ == '__main__':
    if not install_process():
        sys.exit(1)