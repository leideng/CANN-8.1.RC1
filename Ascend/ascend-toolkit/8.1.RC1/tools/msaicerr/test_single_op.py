#!/usr/bin/env python
# coding=utf-8
"""
Function:
The file mainly involves test single op func.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2024
"""

from ms_interface.single_op_test_frame.single_op_case import SingleOpCase

config = {
    "cce_file": "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.cce",
    "bin_path": "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.o",
    "json_path": "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.json",
    "tiling_data": "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_tiling.bin",
    "tiling_key": "0",
    "block_dim": 32,
    "device_id": 0,
    "ffts_addrs_num": 0,
    "input_file_list": [
        "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.input.0.npy",
        "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.input.1.npy",
        "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.input.2.npy"
    ],
    "output_file_list": [
        "/yourpath/te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1.output.0.npy"
    ],
    "kernel_name": "te_gatherv2_8241ce80d37e6d97ac20a118920e96111fd0f4f0877012278629ce7d5d4c7d4b_1"
}
OP_TEST = "single_op"
SingleOpCase.run(config, OP_TEST)
