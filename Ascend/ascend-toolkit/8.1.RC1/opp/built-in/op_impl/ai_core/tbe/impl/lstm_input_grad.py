#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lstm_grad
"""


# 'pylint: disable=unused-argument
# 'pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda
# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments,import-outside-toplevel
def lstm_input_grad(w, init_c, c, dy, dh, dc, i, j,
                    f, o, tanhct, dx,
                    dh_prev, dc_prev, dgate,
                    kernel_name="lstm_input_grad"):
    """
    Parameters
    ----------
    w : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    init_c : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [ND]
    c : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dy : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dh : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dc : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    i : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [ND]
    j : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [ND]
    f:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    o:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    tanhct:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dx:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dh_prev:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dc_prev:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dgate:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    kernel_name : str
        cce kernel name, default value == "lstm_input_grad"
    Returns
    -------
    None
    """
    pass
