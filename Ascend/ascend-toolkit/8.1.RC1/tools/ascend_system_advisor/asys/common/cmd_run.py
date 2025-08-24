#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import platform
import subprocess
import sys

from common.log import log_debug

__all__ = ["run_command", "run_msnpureport_cmd", "check_command", "run_linux_cmd", "popen_run_cmd"]


def get_os_type():
    return platform.system()


def check_command(command):
    os_type = get_os_type()
    if os_type == "Windows":
        cmd = f"where {command}"
    elif os_type == "Linux":
        cmd = f"which {command}"
    else:
        log_debug("Unsupported operating system.")
        return False
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret.returncode == 0


def run_linux_cmd(cmd, cmp_str=""):
    if not isinstance(cmd, str):
        return False
    ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmp_str:
        return ret.stdout.strip().decode() == cmp_str
    if ret.returncode == 0:
        return True
    return False


def run_command(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8',
                         env=os.environ)
    if ret.returncode == 0:
        if ret.stderr != "":
            return "NONE"
        return ret.stdout.strip()
    else:
        ret_err = ret.stderr
        log_debug('run command: {0} failed, ret_code={1}, ret_err={2}'.format(command, ret.returncode, ret_err))
        if "command not found" in ret_err:
            return 'NONE'
        return ret.stderr.strip().replace("\n", "  ")


def run_msnpureport_cmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8',
                         env=os.environ)
    if ret.returncode == 0:
        return True, ret.stdout
    else:
        ret_err = ret.stderr
        log_debug('run command: {0} failed, ret_code={1}, ret_err={2}'.format(command, ret.returncode, ret_err))
        return False, ret.stderr


class _IgnoreStderr:
    def __init__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)

    def __enter__(self):
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)


def popen_run_cmd(command):
    """
    use the os.popen to run the command
    """
    with _IgnoreStderr():
        cmd = os.popen(command)
        ret = cmd.read()
        cmd.close()

    return ret

