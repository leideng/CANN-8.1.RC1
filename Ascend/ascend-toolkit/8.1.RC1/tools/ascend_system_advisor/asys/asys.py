#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import signal
import sys

from common import consts, log_info, log_error, close_log
from common import RetCode, STACKTRACE
from common.task_common import create_out_timestamp_dir
from common import compress_output_dir_tar
from params import ParamDict
from cmdline import CommandLineParser
from config import AsysConfigParser
from collect import AsysCollect, AsysStackTrace
from launch import AsysLaunch
from info import AsysInfo
from diagnose import AsysDiagnose
from health import AsysHealth
from analyze import AsysAnalyze
from config_cmd import AsysConfig

__all__ = ["main"]


def _check_args_duplicate():
    input_args = [arg.split("=")[0] for arg in sys.argv if '-' in arg.split("=")[0]]
    # remove args duplicate
    args_no_duplicate = set(input_args)
    if len(input_args) > len(args_no_duplicate):
        log_error(f"only one of the {list(args_no_duplicate)} args can be specified.")
        return False
    return True


def _execute_cmd():
    task_res = True
    if ParamDict().get_command() == consts.collect_cmd:
        if ParamDict().get_arg("run_mode") == STACKTRACE:
            task_res = AsysStackTrace().collect_stacktrace()
        else:
            asys_collector = AsysCollect()
            task_res = asys_collector.collect()
            asys_collector.clean_work()
    elif ParamDict().get_command() == consts.launch_cmd:
        asys_launcher = AsysLaunch()
        task_res = asys_launcher.launch()
        asys_launcher.clean_work()
    elif ParamDict().get_command() == consts.info_cmd:
        asys_info = AsysInfo()
        task_res = asys_info.run_info()
    elif ParamDict().get_command() == consts.diagnose_cmd:
        asys_diagnoser = AsysDiagnose()
        task_res = asys_diagnoser.diagnose()
    elif ParamDict().get_command() == consts.health_cmd:
        asys_health = AsysHealth()
        task_res = asys_health.health()
    elif ParamDict().get_command() == consts.analyze_cmd:
        asys_analyze = AsysAnalyze()
        task_res = asys_analyze.analyze()
        if not task_res:
            asys_analyze.clean_output()
    elif ParamDict().get_command() == consts.config_cmd:
        task_res = AsysConfig().config()
    return task_res


def main():
    """entrance of Ascend system advisor"""
    # check args duplicate
    if not _check_args_duplicate():
        return False

    # error stack when ctrl c is ignored
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # 1. parse the command line and check args
    asys_parser = CommandLineParser()
    parse_ret = asys_parser.parse()
    if ParamDict().get_command() is None:
        if parse_ret == RetCode.SUCCESS:    # -h, --help, and only asys
            asys_parser.print_help()
            return True
        else:
            log_error("arguments parse failed, asys exit.")
            return False

    # info/diagnose/health close info & warning level log
    if ParamDict().get_command() in [consts.info_cmd, consts.diagnose_cmd, consts.health_cmd] or \
            (ParamDict().get_command() == consts.config_cmd and ParamDict().get_arg("get")):
        close_log()

    env_ret = ParamDict().get_env_type()
    if not env_ret or env_ret == RetCode.FAILED:
        log_error("Failed to obtain the execution environment type.")
        return False
    if env_ret == "RC":
        if ParamDict().get_command() not in [consts.collect_cmd, consts.launch_cmd] or \
                (ParamDict().get_command() == consts.collect_cmd and ParamDict().get_arg("run_mode")):
            log_error("The RC supports the launch command and the collect command without the -r parameter.")
            return False

    # 2. read the config file and load configs
    conf_parser = AsysConfigParser()
    conf_res = conf_parser.parse()
    if not conf_res:
        log_error('configs parse failed, asys exit.')
        return False

    log_info("asys start.")

    if create_out_timestamp_dir() != RetCode.SUCCESS:
        log_error("create asys output directory failed.")
        return False

    # 3. execute the command
    task_res = _execute_cmd()

    log_info("{0} task execute finish.".format(ParamDict().get_command()))

    # 4. Compress the output dir using tar.
    if ParamDict().get_arg("tar") in ["T", "TRUE"]:
        compress_output_dir_tar()

    log_info("asys finish.")
    return task_res


if __name__ == "__main__":
    main()
