#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""
import warnings

from tbe.dsl.static_schedule.elewise_schedule_new import ElewiseSchedule


def get_op_info(outs):
    """
    dfs the compute garph to get the op info, the fomrt as follows:
        op_info
        {
        pattern: "xxx"
        input_tensors : []
        mid_tensors : []
        output_tensors : []
        tensor_map: {input : [outputs]}
        }
    Parameters
    ----------
    outs : the outputs of op

    Returns
    -------
    op_info
    """
    warnings.warn(
        "te.lang.cce.get_op_info is deprecated, please replace it with tbe.dsl.static_schedule.get_op_info",
        DeprecationWarning)
    from tbe.dsl.static_schedule.cce_schedule import get_op_info
    return get_op_info(outs)


def verify_compute_tensor(tensors):
    """
    verify compute tensor by rule:
    rule 1: only one tensor in compute, return False
    rule 2: any compute tensor shall be taggeg with 'elewise_single_cast', if correct return False
    otherwise return True
    tensors: target tensor which needs to verify
    """
    warnings.warn(
        "te.lang.cce.verify_compute_tensor is deprecated, "
        "please replace it with tbe.dsl.static_schedule.verify_compute_tensor",
        DeprecationWarning)
    from tbe.dsl.static_schedule.cce_schedule import verify_compute_tensor
    return verify_compute_tensor(tensors)


def schedule_cce(outs, option=None):
    """
    schedule cce
    """
    warnings.warn(
        "te.lang.cce.schedule_cce is deprecated, please replace it with tbe.dsl.static_schedule.schedule_cce",
        DeprecationWarning)
    from tbe.dsl.static_schedule.cce_schedule import schedule_cce
    return schedule_cce(outs, option)


def check_is_need_cast(out):
    """
    Check if tensor needs to do cast operation

    Parameters
    ----------
    out : output tensor

    Returns
    -------
    Bool : true or false
    """
    warnings.warn(
        "te.lang.cce.check_is_need_cast is deprecated, "
        "please replace it with tbe.dsl.static_schedule.check_is_need_cast",
        DeprecationWarning)
    from tbe.dsl.static_schedule.cce_schedule import check_is_need_cast
    return check_is_need_cast(out)


def cce_build_code(sch, config_map=None):
    """
    API of building or printing lower code, just can be used when device is CCE

    Parameters
    ----------
    sch : tvm.schedule
        schedule to build or to print lower code

    config_map : dict, default is {} and use default configration

        key_words:

            print_ir : if need print lower IR code, default is True

            need_build : if need build, default is True

            name : kernel name, default is cce_op

    Returns
    -------
    None
    """
    from tbe.dsl.static_schedule.cce_schedule import cce_build_code
    return cce_build_code(sch, config_map)
