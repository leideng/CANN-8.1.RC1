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

from dataflow.dataflow import (
    CountBatch,
    finalize,
    FlowFlag,
    FlowGraph,
    FlowInfo,
    FlowData,
    FlowNode,
    Framework,
    FuncProcessPoint,
    GraphProcessPoint,
    FlowGraphProcessPoint,
    init,
    Tensor,
    TensorDesc,
    TimeBatch,
    alloc_tensor,
)
from dataflow.data_type import (
    DT_FLOAT,
    DT_FLOAT16,
    DT_INT8,
    DT_INT16,
    DT_UINT16,
    DT_UINT8,
    DT_INT32,
    DT_INT64,
    DT_UINT32,
    DT_UINT64,
    DT_BOOL,
    DT_DOUBLE,
    DT_STRING,
)
from dataflow.pyflow import pyflow, method
from dataflow.plugin.torch.torch_plugin import npu_model
from dataflow.utils.utils import DfException, DfAbortException, get_running_device_id, get_running_instance_id, \
    get_running_instance_num
from dataflow import utils
from dataflow.utils.msg_type_register import msg_type_register
