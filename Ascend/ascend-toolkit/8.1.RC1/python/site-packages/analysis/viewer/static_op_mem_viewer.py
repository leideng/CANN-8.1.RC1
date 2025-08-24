#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from common_func.ms_constant.str_constant import StrConstant
from common_func.db_name_constant import DBNameConstant
from common_func.msvp_constant import MsvpConstant
from msmodel.add_info.static_op_mem_viewer_model import StaticOpMemViewModel


class StaticOpMemViewer:
    """
    class for get static_op_mem data
    """

    def __init__(self: any, configs: dict, params: dict) -> None:
        self.configs = configs
        self.params = params

    def get_summary_data(self: any) -> tuple:
        """
        to get summary data
        :return:summary data
        """
        model = StaticOpMemViewModel(self.params.get(StrConstant.PARAM_RESULT_DIR), DBNameConstant.DB_STATIC_OP_MEM,
                                     [DBNameConstant.TABLE_STATIC_OP_MEM])
        if not model.check_table():
            return MsvpConstant.MSVP_EMPTY_DATA
        static_op_mem_summary_data = model.get_summary_data()
        if not static_op_mem_summary_data:
            return MsvpConstant.MSVP_EMPTY_DATA
        return self.configs.get(StrConstant.CONFIG_HEADERS), static_op_mem_summary_data, len(static_op_mem_summary_data)
