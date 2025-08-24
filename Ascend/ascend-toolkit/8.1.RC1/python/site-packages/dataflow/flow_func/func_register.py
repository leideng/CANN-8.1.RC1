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


class FlowFuncRegister:
    _name_dict = {}

    @classmethod
    def register_flow_func(cls, model_name, clz_name, func_name, params):
        key = (model_name, clz_name)
        if key not in cls._name_dict:
            cls._name_dict[key] = FlowFuncInfos()
        cls._name_dict[key].add_func_params(func_name, params)

    @classmethod
    def get_flow_func(cls, model_name, clz_name):
        key = (model_name, clz_name)
        if key not in cls._name_dict:
            return None
        else:
            return cls._name_dict[key].get_all_func_params()

    @classmethod
    def register_flow_func_infos(cls, model_name, clz_name, flow_func_infos):
        key = (model_name, clz_name)
        cls._name_dict[key] = flow_func_infos

    @classmethod
    def get_flow_func_infos(cls, model_name, clz_name):
        key = (model_name, clz_name)
        return cls._name_dict.get(key, None)


class FlowFuncInfos:
    def __init__(self):
        self._name_to_params = {}
        self._running_resources_info = []
        self._func_object = None
        self._env_hook_func = None
        self._visible_device_enable = None

    def add_func_params(self, func_name, func_params):
        self._name_to_params[func_name] = func_params

    def get_all_func_params(self):
        return self._name_to_params

    def add_running_resources_info(self, type, num):
        self._running_resources_info.append({"type": type, "num": num})

    def get_running_resources_info(self):
        return self._running_resources_info

    def set_func_object(self, func_object):
        self._func_object = func_object

    def get_func_object(self):
        return self._func_object

    def set_env_hook_func(self, env_hook_func):
        self._env_hook_func = env_hook_func

    def get_env_hook_func(self):
        return self._env_hook_func

    def set_visible_device_enable(self, visible_device_enable):
        self._visible_device_enable = visible_device_enable

    def get_visible_device_enable(self):
        return self._visible_device_enable
