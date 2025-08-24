#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

from common_func.ms_constant.str_constant import StrConstant


def get_task_type(type_num: str) -> str:
    """
    get kernel type
    """
    if type_num in StrConstant.TASK_TYPE_MAPPING:
        kernel_type = StrConstant.TASK_TYPE_MAPPING.get(type_num)
    else:
        kernel_type = type_num
    return kernel_type
