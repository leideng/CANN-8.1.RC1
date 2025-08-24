# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import re


PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")

CONFIG_FILE_MAX_SIZE = 1 * 1024 * 1024 # work for .ini config file
TEXT_FILE_MAX_SIZE = 100 * 1024 * 1024 # work for txt, csv, py
JSON_FILE_MAX_SIZE = 1024 * 1024 * 1024
ONNX_MODEL_MAX_SIZE = 2 * 1024 * 1024 * 1024
TENSOR_MAX_SIZE = 10 * 1024 * 1024 * 1024
MODEL_WEIGHT_MAX_SIZE = 300 * 1024 * 1024 * 1024
DB_MAX_SIZE = 5 * 1024 * 1024 * 1024
INPUT_FILE_MAX_SIZE = 5 * 1024 * 1024 * 1024


EXT_SIZE_MAPPING = {
    '.db': DB_MAX_SIZE,
    '.py': TEXT_FILE_MAX_SIZE,
    ".ini": CONFIG_FILE_MAX_SIZE,
    '.csv': TEXT_FILE_MAX_SIZE,
    '.txt': TEXT_FILE_MAX_SIZE,
    '.pth': MODEL_WEIGHT_MAX_SIZE,
    '.bin': MODEL_WEIGHT_MAX_SIZE,
    '.json': JSON_FILE_MAX_SIZE,
    '.onnx': ONNX_MODEL_MAX_SIZE,
}

