#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import functools
import inspect
import traceback
from dataflow.flow_func.func_register import FlowFuncRegister
from . import flow_func as fw


def proc_wrapper(func_list=None):
    def wrapper(func):
        if func_list is not None:
            module_name = inspect.getmodule(func).__name__
            clz_name = func.__qualname__.split(".")[0]
            FlowFuncRegister.register_flow_func(
                module_name, clz_name, func.__name__, func_list
            )

        @functools.wraps(func)
        def proc_func(self, run_context, input_flow_msgs):
            logger = fw.FlowFuncLogger()
            try:
                py_run_context = fw.MetaRunContext(run_context)
                inputs = [fw.FlowMsg(msg) for msg in input_flow_msgs]
                logger.info("trans to context and inputs success.")
                return func(self, py_run_context, inputs)
            except Exception as e:
                traceback.print_exc()
                logger.error("proc wrapper exception")
                return fw.FLOW_FUNC_FAILED

        return proc_func

    return wrapper


def init_wrapper():
    def wrapper(func):
        @functools.wraps(func)
        def proc_init(self, meta_params):
            logger = fw.FlowFuncLogger()
            try:
                py_meta_params = fw.PyMetaParams(meta_params)
                logger.info("trans to meta parmas wrapper success.")
                return func(self, py_meta_params)
            except Exception as e:
                traceback.print_exc()
                logger.error("init wrapper exception")
                return fw.FLOW_FUNC_FAILED

        return proc_init

    return wrapper
