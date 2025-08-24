#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import sys
import importlib

so_path = os.path.join(os.path.dirname(__file__), "..", "lib64")
sys.path.append(os.path.realpath(so_path))

mspti_C_module = importlib.import_module("mspti_C")
_start = mspti_C_module.start
_stop = mspti_C_module.stop
_flush_all = mspti_C_module.flush_all
_flush_period = mspti_C_module.flush_period

mspti_C_mstx_module = importlib.import_module("mspti_C.mstx")
_mstx_register_cb = mspti_C_mstx_module.registerCB
_mstx_unregister_cb = mspti_C_mstx_module.unregisterCB

mspti_C_kernel_module = importlib.import_module("mspti_C.kernel")
_kernel_register_cb = mspti_C_kernel_module.registerCB
_kernel_unregister_cb = mspti_C_kernel_module.unregisterCB

mspti_C_hccl_module = importlib.import_module("mspti_C.hccl")
_hccl_register_cb = mspti_C_hccl_module.registerCB
_hccl_unregister_cb = mspti_C_hccl_module.unregisterCB