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

import sys
import dataflow.utils.utils as utils


class MsgTypeRegister:
    def __init__(self) -> None:
        self._registered_msg = {65535: "__PickledMsg__"}
        self._registered_clz_to_msg_type = {"__PickledMsg__": 65535}
        self._serialize_func = {65535: self._serialize_with_cloudpickle}
        self._deserialize_func = {65535: self._deserialize_with_cloudpickle}
        self._size_func = {}

    def register(self, msg_type, clz, serialize_func, deserialize_func, size_func=None):
        int_msg_type = int(msg_type)
        if (
            int_msg_type in self._registered_msg
            and self._registered_msg[int_msg_type] != clz
        ):
            raise utils.DfException(
                f"msg_type:{msg_type} has been registered, class:{clz}."
            )
        if (
            clz in self._registered_clz_to_msg_type
            and int_msg_type != self._registered_clz_to_msg_type[clz]
        ):
            raise utils.DfException(
                f"class:{clz} has been registered, "
                f"msg_type:{self._registered_clz_to_msg_type[clz]}."
            )
        self._registered_msg[int_msg_type] = clz
        self._registered_clz_to_msg_type[clz] = int_msg_type
        self._serialize_func[int_msg_type] = serialize_func
        self._deserialize_func[int_msg_type] = deserialize_func
        if size_func is not None:
            self._size_func[int_msg_type] = size_func

    def registered(self, msg_type):
        int_msg_type = int(msg_type)
        return int_msg_type in self._serialize_func

    def get_msg_type(self, clz):
        return self._registered_clz_to_msg_type.get(clz, None)

    def get_serialize_func(self, msg_type):
        int_msg_type = int(msg_type)
        return self._serialize_func.get(int_msg_type, None)

    def get_deserialize_func(self, msg_type):
        int_msg_type = int(msg_type)
        return self._deserialize_func.get(int_msg_type, None)

    def get_size_func(self, msg_type):
        int_msg_type = int(msg_type)
        return self._size_func.get(int_msg_type, None)

    def deserialize_from_file(self, pkl_file, work_path=None):
        if work_path is not None and work_path not in sys.path:
            sys.path.append(work_path)
        import cloudpickle

        with open(pkl_file, "rb") as f:
            return cloudpickle.load(f)

    def _serialize_with_cloudpickle(self, obj):
        import cloudpickle

        return cloudpickle.dumps(obj)

    def _deserialize_with_cloudpickle(self, buffer):
        import cloudpickle

        return cloudpickle.loads(buffer)


msg_type_register = MsgTypeRegister()
