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

from dataflow.flow_func.flow_func import FLOW_FUNC_SUCCESS, FLOW_FUNC_FAILED, FLOW_FUNC_ERR_PARAM_INVALID, \
    FLOW_FUNC_ERR_ATTR_NOT_EXITS, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH, FLOW_FUNC_ERR_TIME_OUT_ERROR, \
    FLOW_FUNC_ERR_DRV_ERROR, FLOW_FUNC_ERR_QUEUE_ERROR, FLOW_FUNC_ERR_MEM_BUF_ERROR, FLOW_FUNC_ERR_EVENT_ERROR, \
    FLOW_FUNC_ERR_USER_DEFINE_START, FLOW_FUNC_ERR_USER_DEFINE_END, FlowFuncLogger, FlowMsg, PyMetaParams, \
    MetaRunContext, MSG_TYPE_RAW_MSG, MSG_TYPE_TENSOR_DATA, FLOW_FLAG_EOS, FLOW_FLAG_SEG, logger, \
    BalanceConfig, AffinityPolicy, MSG_TYPE_PICKLED_MSG, MSG_TYPE_TORCH_TENSOR_MSG, MSG_TYPE_USER_DEFINE_START, \
    FlowMsgQueue, FLOW_FUNC_STATUS_REDEPLOYING, FLOW_FUNC_STATUS_EXIT

from dataflow.flow_func.func_wrapper import proc_wrapper, init_wrapper
from dataflow.flow_func.func_register import FlowFuncRegister, FlowFuncInfos