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

from common import Logger as logger2
from common import rmtree_file

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True

SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DEBUG_KERNEL_PATH = os.path.realpath(os.path.join(SCRIPT_DIR_PATH, ".."))


def uninstall_process():
    logger2.setting_log()
    logger2.info(f"start uninstall ...")
    if not rmtree_file(DEBUG_KERNEL_PATH):
        return False
    logger2.info(f"Uninstallation completed.")
    return True


if __name__ == '__main__':
    if not uninstall_process():
        sys.exit(1)