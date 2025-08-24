#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from common import log_debug
from common import consts
from common import Singleton
from drv import LoadSoType

__all__ = ["ParamDict"]


class ParamDict(metaclass=Singleton):

    def __init__(self):
        self.__command = None
        self.__asys_output_timestamp_dir = None
        self.__args = dict()
        self.__deps = dict()
        self.__ini = dict()
        self.__env_type = None
        self.__task_pid = None

    def get_command(self):
        return self.__command

    @property
    def asys_output_timestamp_dir(self):
        return self.__asys_output_timestamp_dir

    @asys_output_timestamp_dir.setter
    def asys_output_timestamp_dir(self, path):
        self.__asys_output_timestamp_dir = path

    def get_ini(self, key):
        return self.__ini.get(key, False)

    def set_ini(self, ini_name, ini_value):
        self.__ini[ini_name] = ini_value

    def get_arg(self, key):
        if key in self.__args.keys():
            return self.__args.get(key, False)
        return False

    def get_deps(self):
        return self.__deps

    def get_env_type(self):
        if self.__env_type is None:
            self.set_env_type()
        return self.__env_type

    def set_env_type(self, env_type=None):
        if env_type is None:
            self.__env_type = LoadSoType().get_env_type()
        else:
            self.__env_type = env_type

    def set_task_pid(self, pid):
        self.__task_pid = pid

    def get_task_pid(self):
        return self.__task_pid

    def _set_arg_d(self, args):
        if getattr(args, "d") is not None:
            self.__add_arg("device_id", args.d)
            log_debug("set arg: -d=\"{0}\" success.".format(self.__args.get("d")))

    def _set_arg_r(self, args):
        if getattr(args, "r") is not None:
            self.__add_arg("run_mode", args.r)
            log_debug("set arg: -r=\"{0}\" success.".format(self.__args.get("r")))

    def _set_arg_tar(self, args):
        arg_name = "tar"
        if getattr(args, arg_name) is not None:
            self.__add_arg(arg_name, args.tar.upper())
            log_debug("set arg: --output=\"{0}\" success.".format(self.__args.get(arg_name)))

    def _set_arg_symbol_path(self, args):
        arg_name = "symbol_path"
        if getattr(args, arg_name) is not None:
            self.__add_arg(arg_name, args.symbol_path.split(","))
            log_debug("set arg: --symbol_path=\"{0}\" success.".format(self.__args.get(arg_name)))

    def _set_arg_common(self, args, arg_name):
        if getattr(args, arg_name) is not None:
            self.__add_arg(arg_name, eval(f"args.{arg_name}"))
            log_debug(f"set arg: --{arg_name}=\"{self.__args.get(arg_name)}\" success.")

    def set_args(self, args):
        self.__command = args.subparser_name
        _output = "output"

        if self.__command == consts.diagnose_cmd:
            self._set_arg_d(args)
            self._set_arg_r(args)
            self._set_arg_common(args, _output)
            self._set_arg_common(args, "timeout")
            return

        if self.__command == consts.health_cmd:
            self._set_arg_d(args)
            return

        if self.__command == consts.info_cmd:
            self._set_arg_d(args)
            self._set_arg_r(args)
            return

        if self.__command == consts.analyze_cmd:
            self._set_arg_r(args)
            self._set_arg_common(args, "file")
            self._set_arg_common(args, "path")
            self._set_arg_common(args, "core_file")
            self._set_arg_common(args, "exe_file")
            self._set_arg_common(args, "symbol")
            self._set_arg_symbol_path(args)
            self._set_arg_common(args, _output)
            self._set_arg_common(args, "reg")
            return

        if self.__command == consts.config_cmd:
            self._set_arg_d(args)
            self._set_arg_common(args, "get")
            self._set_arg_common(args, "restore")
            self._set_arg_common(args, "stress_detect")
            return

        if self.__command == consts.collect_cmd:
            self._set_arg_common(args, "task_dir")
            self._set_arg_r(args)
            self._set_arg_common(args, "remote")
            self._set_arg_common(args, "all")
            self._set_arg_common(args, "quiet")

        if self.__command == consts.launch_cmd:
            self._set_arg_common(args, "task")

        self._set_arg_common(args, _output)
        self._set_arg_tar(args)

    def set_deps(self, deps):
        for item in deps:
            self.__add_dep(item[0], item[1])

    def __add_arg(self, key, value):
        if key in self.__args.keys():
            return False
        else:
            self.__args[key] = value
            return True

    def __add_dep(self, key, value):
        if key in self.__deps.keys():
            return False
        else:
            self.__deps[key] = value
            return True
