#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import json
import logging

from common_func.constant import Constant


class JsonManager:
    """
    class to manage json operation
    """

    @staticmethod
    def loads(data: any) -> any:
        if not isinstance(data, str):
            logging.error("Data type is not str!")
            return {}
        try:
            return json.loads(data)
        except json.decoder.JSONDecodeError as err:
            logging.error(err, exc_info=Constant.TRACE_BACK_SWITCH)
            return {}
