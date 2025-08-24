"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

pack
"""
from impl.dynamic.concat_v2_d import concat_v2_d


# 'pylint: disable=invalid-name
def pack_tik(x, y, axis, kernel_name="pack"):
    """
    algorithm: pack
    Concatenates tensors along one dimension.
    Parameters
    ----------
    x : A list of `dict`.dict include keys shape and dtype
    y: dict of output_data, dict include keys shape and dtype
    axis : int, in the range [-rank(values)-1, rank(values)]
    kernel_name : cce kernel name, default value is "pack"
    Returns
    -------
    None
    """
    return concat_v2_d(x, y, axis, kernel_name)
