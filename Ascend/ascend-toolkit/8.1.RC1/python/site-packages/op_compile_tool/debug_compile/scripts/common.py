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
import logging
import inspect
from collections import namedtuple
import shutil

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True


class Logger:

    @classmethod
    def setting_log(cls, log_level: str = "INFO") -> None:
        """
        setting the log format and level
        :param: log_level: "DEBUG", "INFO", "WARNING", "ERROR"
        :return: None
        """
        log_format = f"[%(levelname)s] OP_COMPILER_TOOL(%(process)d): %(asctime)s %(message)s"
        if log_level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            print(f"[ERROR]: '{log_level}' is not the correct configuration option")
            log_level = "INFO"
        # 在logging.basicConfig前清理已有 handlers
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        logging.basicConfig(level=log_level.upper(), format=log_format)
        frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        filename, lineno, function = frameinfo.filename, frameinfo.lineno, frameinfo.function
        logging.debug(f"[{filename}:{lineno}] [{function}] setup log level: [{log_level}].")

    @classmethod
    def debug(cls, user_message: str) -> None:
        """
        :param: user_message
        :return: None
        """
        frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        filename, lineno, function = frameinfo.filename, frameinfo.lineno, frameinfo.function
        user_message = f"[{filename}:{lineno}] [{function}] {user_message}"
        logging.debug(user_message)

    @classmethod
    def info(cls, user_message: str) -> None:
        """
        :param: user_message
        :return: None
        """
        frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        filename, lineno, function = frameinfo.filename, frameinfo.lineno, frameinfo.function
        user_message = f"[{filename}:{lineno}] [{function}] {user_message}"
        logging.info(user_message)

    @classmethod
    def warning(cls, user_message: str) -> None:
        """
        :param: user_message
        :return: None
        """
        frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        filename, lineno, function = frameinfo.filename, frameinfo.lineno, frameinfo.function
        user_message = f"[{filename}:{lineno}] [{function}] {user_message}"
        logging.warning(user_message)

    @classmethod
    def error(cls, user_message: str) -> None:
        """
        :param: user_message
        :return: None
        """
        frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        filename, lineno, function = frameinfo.filename, frameinfo.lineno, frameinfo.function
        user_message = f"[{filename}:{lineno}] [{function}] {user_message}"
        logging.error(user_message)

    @classmethod
    def exception(cls, user_message: str) -> None:
        """
        :param: user_message
        :return: None
        """
        frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        filename, lineno, function = frameinfo.filename, frameinfo.lineno, frameinfo.function
        user_message = f"[{filename}:{lineno}] [{function}] {user_message}"
        logging.exception(user_message)


def verify_pathfile_exist_permission(path_file: str, type_: str, modes: str) -> bool:
    """
    verify that the input path exists
    :param key
        path:
        type: file, path or file
        model: e, r, w, x
    :return: True/False
    """
    exist_state = None
    r_permission_state = None
    w_permission_state = None
    x_permission_state = None
    PathfileState = namedtuple("pathfile_state", \
        ["status", "exist_state", "r_permission_state", "w_permission_state", "x_permission_state"])
    try:
        for mode in modes:
            if mode == "e":
                if type_ == "path":
                    exist_state = False if not os.path.isdir(path_file) else True
                elif type_ == "file":
                    exist_state = False if not os.path.isfile(path_file) else True
            elif mode == "r":
                r_permission_state = True if os.access(path_file, os.R_OK) else False
            elif mode == "w":
                w_permission_state = True if os.access(path_file, os.W_OK) else False
            elif mode == "x":
                x_permission_state = True if os.access(path_file, os.X_OK) else False
    except Exception as e:
        Logger.exception(f"failed to verify file or dir attributes., {e}")
        return PathfileState(False, exist_state, r_permission_state, w_permission_state, x_permission_state)

    return PathfileState(True, exist_state, r_permission_state, w_permission_state, x_permission_state)


def get_opp_path() -> str:
    """
    get opp path
    :return: True/False, opp_path
    """
    opp_path = None
    try:
        opp_path = os.environ["ASCEND_OPP_PATH"]
        opp_path = os.path.realpath(opp_path)
        outputpath_state = verify_pathfile_exist_permission(opp_path, "path", "erwx")
        if not outputpath_state.exist_state:
            Logger.error(f"[{path}] path does not exist. please check.")
            return False, opp_path
        if not outputpath_state.w_permission_state or not outputpath_state.r_permission_state:
            Logger.error(f"[{path}] do not have the read or write permission., please setting.")
            return False, opp_path
        Logger.debug("obtaining ASCEND_OPP_PATH succeeded.")
    except Exception as e:
        Logger.exception(f"failed to obtain ASCEND_OPP_PATH, please setting environment variable.")
        return False, opp_path
    return True , opp_path


def rmtree_file(path: str):
    """
    :param path: 路径
    :return: True/False
    """
    try:
        shutil.rmtree(path)
        Logger.info(f"deleting succeeded, [{path}]")
    except Exception as e:
        Logger.exception(f"failed to delete [{path}] files, {e}")
        return False
    return True