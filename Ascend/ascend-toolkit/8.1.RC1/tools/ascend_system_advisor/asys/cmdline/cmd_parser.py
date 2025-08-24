#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import argparse
import textwrap
import enum

from params import ParamDict
from common.const import RetCode, consts
from cmdline.arg_checker import ArgChecker

KEY_NAME = "name"
KEY_TYPE = "type"
KEY_HELP = "help"
KEY_CHECKER = "checker"
KEY_REQUIRED = "required"
KEY_ARGS = "args"
KEY_CHOICES = "choices"
KEY_METAVAR = "metavar"
KEY_ACTION = "action"

OPTIONAL_Y = "\033[33m<Optional>\033[0m"  # yellow
POSITIONAL_R = "\033[31m<Positional>\033[0m"  # red


class Arg(enum.Enum):
    '''The support arg.'''
    TASK_DIR = {
        KEY_NAME: "task_dir", KEY_CHECKER: ArgChecker.DIR_EXIST, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the directory for collecting operator build files, GE dump graphs, "
                  "and TF Adapter dump graphs. If task_dir is not set, these files are not collected by default.",
    }
    TASK = {
        KEY_NAME: "task", KEY_CHECKER: ArgChecker.EXECUTABLE, KEY_REQUIRED: True, KEY_METAVAR: " ",
        KEY_HELP: f"{POSITIONAL_R} Specifies the execution command for the service. "
                  "It collects maintenance and debugging information during command execution."
    }
    OUTPUT = {
        KEY_NAME: "output", KEY_CHECKER: ArgChecker.DIR_CREATE, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the flush path of the command execution result, Default: current dir."
    }
    TAR = {
        KEY_NAME: "tar", KEY_CHECKER: ArgChecker.TAR_CHECK, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies whether to compress the asys result directory into a tar.gz file."
                  " The original directory is not retained after compression. No compression by default."
    }
    COLLECT_RUN = {
        KEY_NAME: "r", KEY_CHECKER: None, "required": False, KEY_CHOICES: ['stacktrace'],
        KEY_HELP: f"{OPTIONAL_Y} Specifies the collect logs mode, this parameter must be used together with '--remote'"
                  " and '--all'. It can be set to 'stacktrace' (send signal to the process specified by remote, "
                  "and generating the stackcore file). "
                  "If r is not set, collects existing maintenance and debugging information in the environment."
    }
    REMOTE = {
        KEY_NAME: "remote", KEY_TYPE: int, KEY_CHECKER: None, "required": False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the ID of the process that receives signal, "
                  "this parameter must be used together with '-r=stacktrace'."
    }
    ALL = {
        KEY_NAME: "all", KEY_CHECKER: None, "required": False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the stackcore files for all tasks, this parameter must be used together with"
                  " '-r=stacktrace'."
    }
    QUIET = {
        KEY_NAME: "quiet", KEY_CHECKER: None, "required": False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Disable the interaction function during stack information export, "
                  "this parameter must be used together with '-r=stacktrace'."
    }

    DEVICE = {
        KEY_NAME: "d", KEY_TYPE: int, KEY_CHECKER: ArgChecker.DEVICE_ID, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the ID of the device for command execution."
    }

    DIS_RUN = {
        KEY_NAME: "r", KEY_CHECKER: None, "required": True, KEY_CHOICES: ['stress_detect', 'hbm_detect', 'cpu_detect'],
        KEY_HELP: f"{POSITIONAL_R} Specifies the hardware detection mode. It can be set to 'stress_detect' (AI Core "
                  "stress test), 'hbm_detect' (HBM detection) or 'cpu_detect' (CPU detection)."
    }
    TIMEOUT = {
        KEY_NAME: "timeout", KEY_TYPE: int, KEY_CHECKER: None, "required": False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the detection duration, in seconds. "
                  "In HBM detection mode, value range: [0, 604800]. In CPU detection mode, value range: [1, 604800]. "
                  "If this argument is not specified, the default 600s is used."
    }

    INFO_RUN = {
        KEY_NAME: "r", KEY_CHECKER: None, KEY_REQUIRED: True,
        KEY_CHOICES: ['hardware', 'software', 'status'],
        KEY_HELP: f"{POSITIONAL_R} Specifies the type of information to be collected."
                  " It can be set to 'status' (device information), 'software' (software information of the host), "
                  "or 'hardware' (hardware information of the host and device)."
    }

    ANALYZE_RUN = {
        KEY_NAME: "r", KEY_CHECKER: None, KEY_REQUIRED: True,
        KEY_CHOICES: ["trace", "coredump", "stackcore"],
        KEY_HELP: f"{POSITIONAL_R} Specifies the type of data to be analyzed. It can be set to 'trace' "
                  "(trace binary file), 'coredump' (system core file), or 'stackcore' (stackcore file)."
    }
    FILE = {
        KEY_NAME: "file", KEY_CHECKER: ArgChecker.FILE_PATH_EXIST_R, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{POSITIONAL_R} Specifies the single file to be analyzed. This argument is valid only for 'trace' "
                  "and 'stackcore'. Use either this argument or '--path'."
    }
    PATH = {
        KEY_NAME: "path", KEY_CHECKER: ArgChecker.FILE_PATH_EXIST_R, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{POSITIONAL_R} Specifies the path to be analyzed. This argument is valid only for 'trace' and "
                  "'stackcore'. Use either this argument or '--file'."
    }
    EXE_FILE = {
        KEY_NAME: "exe_file", KEY_CHECKER: None, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{POSITIONAL_R} Specifies the executable file to be debugged. "
                  "This argument is valid only for 'coredump'."
    }
    CORE_FILE = {
        KEY_NAME: "core_file", KEY_CHECKER: ArgChecker.CORE_FILE, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{POSITIONAL_R} Specifies the core file to be debugged. This argument is valid only for 'coredump'."
    }
    SYMBOL = {
        KEY_NAME: "symbol", KEY_TYPE: int, KEY_CHECKER: None, KEY_REQUIRED: False, KEY_CHOICES: [0, 1],
        KEY_HELP: f"{OPTIONAL_Y} Specifies whether to retain the stack frame information that fails to be analyzed "
                  "in the result (represented by double questions marks '??'). "
                  "This argument is valid only for 'coredump'. Defaults to 0, indicating not to retain."
    }
    SYMBOL_PATH = {
        KEY_NAME: "symbol_path", KEY_CHECKER: ArgChecker.SYMBOL_PATH, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Specifies the path of executable files and dependent dynamic library files. "
                  "Subpaths are not searched. This argument is valid only for 'stackcore'. "
                  "Defaults to the dynamic library path in the stackcore file."
    }
    REG = {
        KEY_NAME: "reg", KEY_TYPE: int, KEY_CHECKER: None, KEY_REQUIRED: False, KEY_CHOICES: [0, 1, 2],
        KEY_HELP: f"{OPTIONAL_Y} Specifies the mode of adding register data for analysis. "
                  "0: not add; 1: add only for threads; 2: add for all stack frames. Defaults to 0."
    }

    GET = {
        KEY_NAME: "get", KEY_CHECKER: None, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Gets the configuration. Use either this argument or '--restore'."
    }
    RESTORE = {
        KEY_NAME: "restore", KEY_CHECKER: None, KEY_REQUIRED: False, KEY_METAVAR: " ",
        KEY_HELP: f"{OPTIONAL_Y} Restores the configuration. Use either this argument or '--get'."
    }
    STRESS_DETECT = {
        KEY_NAME: "stress_detect", KEY_CHECKER: None, KEY_REQUIRED: True, KEY_METAVAR: " ",
        KEY_HELP: f"{POSITIONAL_R} Specifies the configuration options to be queried or restored, "
                  "indicating the configurations related to the pressure test."
    }


class Command(enum.Enum):
    '''The support command.'''
    COLLECT = {
        KEY_NAME: "collect",
        KEY_ARGS: [Arg.TASK_DIR, Arg.OUTPUT, Arg.TAR, Arg.COLLECT_RUN, Arg.REMOTE, Arg.ALL, Arg.QUIET],
        KEY_HELP: "Collects existing maintenance and debugging information in the environment, "
                  "or export stacktrace information in real time."
    }
    LAUNCH = {
        KEY_NAME: "launch",
        KEY_ARGS: [Arg.TASK, Arg.OUTPUT, Arg.TAR],
        KEY_HELP: "Executes the script of task parameters, and collects the maintenance "
                  "and debugging information during the script execution."
    }
    DIAGNOSE = {
        KEY_NAME: "diagnose",
        KEY_ARGS: [Arg.DEVICE, Arg.DIS_RUN, Arg.TIMEOUT, Arg.OUTPUT],
        KEY_HELP: "Diagnoses the hardware status of the device, only 910B and 910_93 are supported."
    }
    HEALTH = {
        KEY_NAME: "health",
        KEY_ARGS: [Arg.DEVICE],
        KEY_HELP: "Diagnoses the health status of the device."
    }
    INFO = {
        KEY_NAME: "info",
        KEY_ARGS: [Arg.DEVICE, Arg.INFO_RUN],
        KEY_HELP: "Collects the software and hardware information of the host and device."
    }
    ANALYZE = {
        KEY_NAME: "analyze",
        KEY_ARGS: [Arg.ANALYZE_RUN, Arg.FILE, Arg.PATH, Arg.EXE_FILE, Arg.CORE_FILE, Arg.SYMBOL, Arg.SYMBOL_PATH,
                   Arg.REG, Arg.OUTPUT],
        KEY_HELP: "Analyzes the trace binary file, core file, and stackcore file."
    }
    CONFIG = {
        KEY_NAME: "config",
        KEY_ARGS: [Arg.DEVICE, Arg.GET, Arg.RESTORE, Arg.STRESS_DETECT],
        KEY_HELP: "Gets or restores configuration information."
    }


class CommandLineParser:
    """
    The definition of command line parser.

    Args:
        args_parsed: The result after parsed.
        parser: The parser to parse the command line.
    """

    def __init__(self):
        description_msg = textwrap.dedent('''\
            command help:
                asys {command} [-h, --help]
                ''')
        self.parser = argparse.ArgumentParser(prog="asys", formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description=description_msg)
        subparsers = self.parser.add_subparsers(dest='subparser_name', help='asys supported commands')

        # Config the parser from Command and Args
        for cmd in Command:
            cmd_conf = cmd.value
            # analyze, diagnose only support EP
            if ParamDict().get_env_type() == "RC" and cmd_conf[KEY_NAME] not in [consts.collect_cmd, consts.launch_cmd]:
                continue
            parser = subparsers.add_parser(cmd_conf[KEY_NAME], help=cmd_conf[KEY_HELP], allow_abbrev=False)
            if cmd_conf[KEY_NAME] == consts.config_cmd:
                self.__set_config_cmd_parser(parser, cmd_conf)
                continue
            if cmd_conf[KEY_NAME] == consts.analyze_cmd:
                self.__set_analyze_cmd_parser(parser, cmd_conf)
                continue

            supported_args = cmd_conf[KEY_ARGS]
            for arg in supported_args:
                arg_conf = arg.value
                if arg_conf.get(KEY_NAME) in ['d', 'r']:
                    arg_name = "-" + arg_conf[KEY_NAME]
                else:
                    arg_name = "--" + arg_conf[KEY_NAME]
                _metavar = " "
                if arg_conf.get(KEY_CHOICES):
                    _metavar = None
                if arg_conf[KEY_NAME] in ["all", "quiet"]:
                    parser.add_argument(
                        arg_name, required=arg_conf[KEY_REQUIRED], action="store_true", help=arg_conf[KEY_HELP]
                    )
                else:
                    parser.add_argument(
                        arg_name, type=arg_conf.get(KEY_TYPE, str), required=arg_conf[KEY_REQUIRED],
                        choices=arg_conf.get(KEY_CHOICES), help=arg_conf[KEY_HELP], metavar=arg_conf.get(KEY_METAVAR)
                    )

    @staticmethod
    def __set_config_cmd_parser(parser, cmd_conf):
        group = parser.add_mutually_exclusive_group(required=False)
        supported_args = cmd_conf[KEY_ARGS]
        for arg in supported_args:
            arg_conf = arg.value
            arg_name = "--" + arg_conf[KEY_NAME]
            if arg_conf.get(KEY_NAME) == 'd':
                arg_name = "-" + arg_conf[KEY_NAME]
                parser.add_argument(
                    arg_name, type=arg_conf.get(KEY_TYPE, str), required=arg_conf[KEY_REQUIRED],
                    choices=arg_conf.get(KEY_CHOICES), help=arg_conf[KEY_HELP], metavar=arg_conf.get(KEY_METAVAR)
                )
                continue

            if arg_conf.get(KEY_NAME) in ["get", "restore"]:
                group.add_argument(arg_name, required=arg_conf[KEY_REQUIRED], action="store_true",
                                   help=arg_conf[KEY_HELP])
            else:
                parser.add_argument(arg_name, required=arg_conf[KEY_REQUIRED], action="store_true",
                                    help=arg_conf[KEY_HELP])

    @staticmethod
    def __set_analyze_cmd_parser(parser, cmd_conf):
        group = parser.add_mutually_exclusive_group(required=False)
        supported_args = cmd_conf[KEY_ARGS]
        for arg in supported_args:
            arg_conf = arg.value
            arg_name = "--" + arg_conf[KEY_NAME]
            if arg_conf.get(KEY_NAME) == 'r':
                arg_name = "-" + arg_conf[KEY_NAME]

            if arg_conf.get(KEY_NAME) in ["file", 'path']:
                group.add_argument(arg_name, type=arg_conf.get(KEY_TYPE, str), required=arg_conf[KEY_REQUIRED],
                                   help=arg_conf[KEY_HELP], metavar=arg_conf.get(KEY_METAVAR))
            else:
                parser.add_argument(
                    arg_name, type=arg_conf.get(KEY_TYPE, str), required=arg_conf[KEY_REQUIRED],
                    choices=arg_conf.get(KEY_CHOICES), help=arg_conf[KEY_HELP], metavar=arg_conf.get(KEY_METAVAR)
                )

    def print_help(self):
        """Print the help information generated by parser"""
        self.parser.print_help()

    @classmethod
    def check_arg_with_checker(cls, arg_name, arg_val, checker):
        """
        Check arg with checker.

        Args:
            arg_val: The value of arg to check
            checker: The check function to use

        Returns:
            RetCode: return code (SUCCESS:0, FAILED:1)
        """
        if checker is None:
            return RetCode.SUCCESS
        return checker(arg_name, arg_val)

    @classmethod
    def match_command(cls, cmd):
        """
        Match input command to Enum cmd type.

        Args:
            cmd: Input command

        Returns:
            command: Enum cmd type
        """
        for command in Command:
            command_conf = command.value
            if command_conf[KEY_NAME] == cmd:
                return command
        return None

    @classmethod
    def check_args(cls, args):
        """
        Check args according to command type.

        Args:
            args: The namesapce returned by parse_args

        Returns:
            RetCode: return code (SUCCESS:0, FAILED:1)
        """
        input_cmd = args.subparser_name
        command = CommandLineParser.match_command(input_cmd)
        if not command:
            return RetCode.FAILED
        supported_args = command.value[KEY_ARGS]
        for support_arg in supported_args:
            arg_info = support_arg.value
            arg_val = getattr(args, arg_info[KEY_NAME])
            if arg_val is None:
                continue  # this arg is optional and not set, check next arg
            checker = arg_info[KEY_CHECKER]
            ret = CommandLineParser.check_arg_with_checker(arg_info[KEY_NAME], arg_val, checker)
            if ret != RetCode.SUCCESS:
                return RetCode.FAILED
        return RetCode.SUCCESS

    def parse(self):
        """
        Parse the command and args from cmd line.

        Returns:
            RetCode: return code (SUCCESS:0, FAILED:1)
        """
        args = self.parser.parse_args()
        if args.subparser_name is None:  # -h, --help, and only asys
            return RetCode.SUCCESS
        if CommandLineParser.check_args(args) == RetCode.FAILED:
            return RetCode.FAILED
        ParamDict().set_args(args)
        return RetCode.SUCCESS
