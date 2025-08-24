#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import logging
import os

from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.path_manager import PathManager
from common_func.profiling_scene import ProfilingScene


class TrailingCalculator:
    """
    calculate slow node message.
    """
    SLOW_THRESHOLD = 20

    def __init__(self: any, path_list: list) -> None:
        self._cluster_list = path_list
        self.trailing_dict = {}

    @staticmethod
    def get_step_data(data_path: str) -> list:
        sql_path = PathManager.get_db_path(data_path, DBNameConstant.DB_TRACE)
        conn, curs = DBManager.check_connect_db_path(sql_path)
        avg_time = []
        if DBManager.check_tables_in_db(sql_path, DBNameConstant.TABLE_TRAINING_TRACE):
            avg_time = DBManager.fetch_all_data(curs, "select avg(data_aug_bound) from {}".format(
                DBNameConstant.TABLE_TRAINING_TRACE))
        return avg_time

    def run(self: any) -> dict:
        """
        entrance for calculating slow node
        :return: slow node list
        """
        for data_path in self._cluster_list:
            ProfilingScene().init(data_path)
            if ProfilingScene().is_operator():
                continue
            self.calculate(data_path)
        return self.calculate_slow_node()

    def calculate_slow_node(self: any) -> dict:
        slow_node_dict = {'Slow Node': []}
        if not self.trailing_dict:
            return slow_node_dict
        try:
            sum_time = sum(self.trailing_dict.values())
        except TypeError as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
            return slow_node_dict
        if not sum_time:
            return slow_node_dict
        try:
            avg_bound = round(sum_time / len(self.trailing_dict.values()), NumberConstant.ROUND_TWO_DECIMAL)
        except (ZeroDivisionError, TypeError) as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
            return slow_node_dict
        try:
            for key, value in self.trailing_dict.items():
                ratio = round(NumberConstant.PERCENTAGE * (value - avg_bound) / avg_bound,
                              NumberConstant.ROUND_TWO_DECIMAL)
                if ratio > self.SLOW_THRESHOLD:
                    slow_node = 'Node: {0}, with a data augmentation bound duration of {1} ns on average, ' \
                                'which is {2}% higher than the average duration consumed by nodes.\t' \
                        .format(os.path.basename(key), str(round(value, NumberConstant.ROUND_TWO_DECIMAL)),
                                str(ratio))
                    slow_node_dict.setdefault('Slow Node', []).append(slow_node)
        except (ZeroDivisionError, TypeError) as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
            return slow_node_dict
        return slow_node_dict

    def calculate(self: any, data_path) -> None:
        """
        get and update data
        :return: None
        """
        time = self.get_step_data(data_path)
        if time:
            self.trailing_dict[data_path] = time[0][0]
