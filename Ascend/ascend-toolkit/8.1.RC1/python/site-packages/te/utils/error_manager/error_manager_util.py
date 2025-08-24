#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
error manager util
"""
import json
import os
import warnings

STACKLEVEL_FOR_ERROR_MESSAGE = 2


def get_error_message(args):
    """
    :param args: dict
        keys in dict must be in accordance with xlsx
    :return: string
            formatted message
    """
    warnings.warn("te.utils.error_manager is expired, please replace it with tbe.common.utils.errormgr",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_ERROR_MESSAGE)
    from tbe.common.utils.errormgr import get_error_message
    return get_error_message(args)



def raise_runtime_error(dict_args):
    """
    raise runtime error
    :param dict_args: error message dict
    """
    warnings.warn("te.utils.error_manager is expired, please replace it with tbe.common.utils.errormgr",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_ERROR_MESSAGE)
    from tbe.common.utils.errormgr import raise_runtime_error
    return raise_runtime_error(dict_args)


def raise_runtime_error_cube(args_dict, msg):
    """
    raise runtime error
    :param args_dict: input dict
    :param msg: error message
    """
    warnings.warn("te.utils.error_manager is expired, please replace it with tbe.common.utils.errormgr",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_ERROR_MESSAGE)
    from tbe.common.utils.errormgr import raise_runtime_error_cube
    return raise_runtime_error_cube(args_dict, msg)
