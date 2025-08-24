#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
from typing import NamedTuple

from params import ParamDict
from common import log_debug, log_warning, log_error
from common import FileOperate as f
from common import get_project_conf
from common.const import RetCode

__all__ = ["AsysConfigParser"]


class IniConfItem(NamedTuple):
    para_name : str
    conf_val_map : dict
    default_val : str


ASYS_INI_VALUE_MAP = {
    "graph" : IniConfItem(
        "graph", {"TRUE":"1", "FALSE":"0"}, "1"
    ),
    "ops" : IniConfItem(
        "ops", {"TRUE":"1", "FALSE":"0"}, "1"
    ),
    "dump_ge_graph" : IniConfItem(
        "DUMP_GE_GRAPH", {"1":"1", "2":"2", "3":"3"}, "2"
    ),
    "dump_graph_level" : IniConfItem(
        "DUMP_GRAPH_LEVEL", {"1":"1", "2":"2", "3":"3"}, "2"
    ),
    "log_level" : IniConfItem(
        "ASCEND_GLOBAL_LOG_LEVEL", {"DEBUG":"0", "INFO":"1", "WARNING":"2", "ERROR":"3", "NULL":"4"}, "1"
    ),
    "log_event_enable" : IniConfItem(
        "ASCEND_GLOBAL_EVENT_ENABLE", {"FALSE":"0", "TRUE":"1"}, "1"
    ),
    "log_print_to_stdout" : IniConfItem(
        "ASCEND_SLOG_PRINT_TO_STDOUT", {"FALSE":"0", "TRUE":"1"}, "0"
    ),
}


class AsysConfigParser:

    def __init__(self):
        self.dep_file_path = os.path.join(get_project_conf(), "dependent_package.csv")
        self.ini_file_path = os.path.join(get_project_conf(), "asys.ini")

    def __parse_deps(self):
        dep_info = f.read_file(self.dep_file_path)
        if not dep_info:
            log_error("read the dependencies file: \"{}\" failed.".format(self.dep_file_path))
            return RetCode.READ_FILE_FAILED

        ParamDict().set_deps(dep_info)
        return RetCode.SUCCESS

    def __parse_ini(self):
        ini_parser = f.read_file(self.ini_file_path)
        if not ini_parser:
            log_error("read the config file: \"{}\" failed.".format(self.ini_file_path))
            return RetCode.READ_FILE_FAILED

        command = ParamDict().get_command()
        if command not in ini_parser.sections():
            log_debug("no ini conf items for command: \"{}\".".format(command))
            return RetCode.SUCCESS
        k_v_pairs = ini_parser.items(command)

        for conf_k, conf_v in k_v_pairs:
            info_item = ASYS_INI_VALUE_MAP.get(conf_k)
            if info_item is None:
                log_warning("ini conf item: \"{}\" is not available.".format(conf_k))
                continue

            ini_name = info_item.para_name
            ini_value = info_item.conf_val_map.get(conf_v)
            if ini_value is None:
                ini_value = info_item.default_val
                log_warning("ini conf value: \"{}\" is not in available range, use dafault value: \"{}\"" \
                            .format(conf_v, ini_value))
            ParamDict().set_ini(ini_name, ini_value)

        return RetCode.SUCCESS

    def parse(self):
        if self.__parse_deps() != RetCode.SUCCESS:
            return False
        if self.__parse_ini() != RetCode.SUCCESS:
            return False
        return True


