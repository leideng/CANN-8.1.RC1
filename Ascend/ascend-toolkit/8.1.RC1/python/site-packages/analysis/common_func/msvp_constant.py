#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2020. All rights reserved.

import os


class MsvpConstant:
    """
    msvp constant
    """
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "msconfig")
    # msvp return empty info
    MSVP_EMPTY_DATA = ([], [], 0)
    EMPTY_DICT = {}
    EMPTY_LIST = []
    EMPTY_TUPLE = ()

    @property
    def msvp_empty_data(self: any) -> tuple:
        """
        default empty data stuct
        :return:
        """
        return self.MSVP_EMPTY_DATA

    @property
    def empty_dict(self: any) -> dict:
        """
        empty dict
        :return:
        """
        return self.EMPTY_DICT
