#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import enum
import re
from common.cmd_run import check_command
from common.const import RetCode
from common import log_error
from common import FileOperate as f
from common.const import DEVICE_ID_MIN, DEVICE_ID_MAX


__all__ = ["ArgChecker", "check_arg_exist_dir", "check_arg_executable", "check_arg_create_dir"]


def empty_str_check(arg_name, arg_val):
    if arg_val == "":
        log_error("argument \"{0}\" can not be empty string, argument value: \"{1}\"".format(arg_name, arg_val))
        return RetCode.ARG_EMPTY_STRING
    return RetCode.SUCCESS


def space_check(arg_name, arg_val):
    if " " in arg_val:
        log_error("argument \"{0}\" can not have space character, argument value: \"{1}\"".format(arg_name, arg_val))
        return RetCode.ARG_SAPCE_STRING
    return RetCode.SUCCESS


def ill_char_check(arg_name, arg_val):
    rex = re.search("[\\\\^$%&*+|#~`=<>,?!:;\'\"\\[\\]\\(\\)\\{\\}]", arg_val)
    if rex:
        log_error("argument \"{0}\" can not have illegal character: \"{1}\", argument value: \"{2}\"" \
                  .format(arg_name, rex.group(), arg_val))
        return RetCode.ARG_ILLEGAL_STRING
    return RetCode.SUCCESS


def path_str_check(arg_name, arg_val):
    if empty_str_check(arg_name, arg_val) != RetCode.SUCCESS:
        return RetCode.ARG_EMPTY_STRING
    if space_check(arg_name, arg_val) != RetCode.SUCCESS:
        return RetCode.ARG_SAPCE_STRING
    if ill_char_check(arg_name, arg_val) != RetCode.SUCCESS:
        return RetCode.ARG_ILLEGAL_STRING
    return RetCode.SUCCESS


def check_arg_exist_dir(arg_name, arg_val):
    """
    Check if arg_val is an exist dir.

    Args:
        arg_val: The value of arg to check

    Returns:
        RetCode: return code (SUCCESS:0, FAILED:1)
    """
    ret = path_str_check(arg_name, arg_val)
    if ret != RetCode.SUCCESS:
        return ret
    if not os.path.isdir(arg_val):
        log_error("argument \"{0}\" is not an exist directory, argument value: \"{1}\"".format(arg_name, arg_val))
        return RetCode.ARG_NO_EXIST_DIR
    return RetCode.SUCCESS


def check_arg_executable(arg_name, arg_val):
    """
    Check if arg_val is an executable command.

    Args:
        arg_val: The value of arg to check

    Returns:
        RetCode: return code (SUCCESS:0, FAILED:1)
    """
    if empty_str_check(arg_name, arg_val.strip()) != RetCode.SUCCESS:
        return RetCode.ARG_EMPTY_STRING

    check_exe = re.compile(r"sh|.*/sh|bash|.*/bash|python[0-9.]*|.*/python[0-9.]*")
    if check_exe.match(arg_val.strip()) and check_exe.match(arg_val.strip()).group() == arg_val.strip():
        log_error(f"argument \"{arg_name}\" no executable script, argument value: \"{arg_val}\"")
        return RetCode.ARG_NO_EXECUTABLE

    check_exe = re.compile(r"sh |.*/sh |bash |.*/bash |python[0-9.]* |.*/python[0-9.]* ")
    if not check_exe.match(arg_val.lstrip()):
        return RetCode.SUCCESS

    check_script = re.compile(
        r"sh .*?\.sh|.*?/sh .*?\.sh|bash .*?\.sh|.*?/bash .*?\.sh|"
        r"sh .*?\.bash|.*?/sh .*?\.bash|bash .*?\.bash|.*?/bash .*?\.bash|"
        r"python[0-9.]*? .*?\.py|.*?/python[0-9.]*? .*?\.py"
    )
    if not check_script.match(arg_val.strip()):
        log_error(f"argument \"{arg_name}\" no executable script, argument value: \"{arg_val}\"")
        return RetCode.ARG_NO_EXECUTABLE
    return RetCode.SUCCESS


def check_arg_create_dir(arg_name, arg_val):
    """
    Check if arg_value is a directory string.
    If arg_val is not exist, it will find its parent dir and check.

    Args:
        arg_val: The value of arg to check

    Returns:
        RetCode: return code (SUCCESS:0, FAILED:1)
    """
    def create_dir_reverse_scan(scan_path):
        while scan_path != "/":
            if os.access(scan_path, os.F_OK):
                if os.access(scan_path, os.W_OK):
                    f.create_dir(arg_val)
                    return RetCode.SUCCESS
                else:
                    return RetCode.ARG_CREATE_DIR_FAILED
            scan_path = os.path.dirname(scan_path)
        return RetCode.ARG_CREATE_DIR_FAILED

    ret = path_str_check(arg_name, arg_val)
    if ret != RetCode.SUCCESS:
        return ret

    arg_val = os.path.abspath(arg_val)
    if os.path.exists(arg_val):
        if not os.path.isdir(arg_val):
            log_error("argument \"{0}\" is existed but not a directory, argument value: \"{1}\"".
                      format(arg_name, arg_val))
            return RetCode.ARG_NO_EXIST_DIR
        if not os.access(arg_val, os.W_OK):
            log_error("argument \"{0}\" is not permissibale to write, argument value: \"{1}\"".
                      format(arg_name, arg_val))
            return RetCode.ARG_NO_WRITABLE_PERMISSION
        return RetCode.SUCCESS
    else:
        scan_path = os.path.dirname(arg_val)
        ret = create_dir_reverse_scan(scan_path)
        if ret != RetCode.SUCCESS:
            log_error("argument \"{0}\" is not permissibale to create, argument value: \"{1}\"".
                      format(arg_name, arg_val))
        return ret


def check_arg_tar(arg_name, arg_val):
    if arg_val.upper() not in ["F", "T", "FALSE", "TRUE"]:
        log_error("argument \"{0}\" should be in [\"F\", \"T\", \"False\", \"True\"], input: \"{1}\"".
                  format(arg_name, arg_val))
        return RetCode.FAILED
    return RetCode.SUCCESS


def check_arg_device_id(arg_name, arg_val):
    if arg_val < DEVICE_ID_MIN or arg_val > DEVICE_ID_MAX:
        log_error("argument \"{}\" value is range of [{}, {}], input: \"{}\"".format(
            arg_name, DEVICE_ID_MIN, DEVICE_ID_MAX, arg_val))
        return RetCode.FAILED
    return RetCode.SUCCESS


def check_arg_exist_or_read_permissibale(arg_name, arg_val):
    """
       Check if arg_value is a directory string.
       If arg_val is not exist,
       If arg_val is not read permissibale

    Args:
        arg_val: The value of arg to check

    Returns:
        RetCode: return code (SUCCESS:0, FAILED:1)
    """

    ret = path_str_check(arg_name, arg_val)
    if ret != RetCode.SUCCESS:
        return ret
    arg_val = os.path.abspath(arg_val)
    if os.path.exists(arg_val):
        if arg_name == "file" and not os.path.isfile(arg_val):
            log_error(f"{arg_val} is not a file.")
            return RetCode.FAILED
        if arg_name == "path" and not os.path.isdir(arg_val):
            log_error(f"{arg_val} is not a directory.")
            return RetCode.FAILED
        if not os.access(arg_val, os.R_OK):
            log_error("argument \"{0}\" is not permissibale to read, argument value: \"{1}\"".
                      format(arg_name, arg_val))
            return RetCode.ARG_NO_WRITABLE_PERMISSION
        return RetCode.SUCCESS
    else:
        log_error(f"{arg_val}: no such file or directory.")
        return RetCode.FAILED


def check_core_file(arg_name, arg_val):
    if not (f.check_exists(arg_val) and f.check_file(arg_val)):
        log_error(f"{arg_val} is not a file or does not exist.")
        return RetCode.FAILED
    return RetCode.SUCCESS


def check_symbol_path(arg_name, arg_val):
    arg_list = arg_val.split(",")
    for dir in arg_list:
        if f.check_valid_dir(dir):
            return RetCode.SUCCESS
    log_error(f"Check whether the {arg_name} parameter is a directory and whether the user has the read permission.")
    return RetCode.FAILED


class ArgChecker(enum.Enum):
    '''The arg type for check.'''
    DIR_EXIST = check_arg_exist_dir
    DIR_CREATE = check_arg_create_dir
    EXECUTABLE = check_arg_executable
    TAR_CHECK = check_arg_tar
    DEVICE_ID = check_arg_device_id
    FILE_PATH_EXIST_R = check_arg_exist_or_read_permissibale
    CORE_FILE = check_core_file
    SYMBOL_PATH = check_symbol_path
