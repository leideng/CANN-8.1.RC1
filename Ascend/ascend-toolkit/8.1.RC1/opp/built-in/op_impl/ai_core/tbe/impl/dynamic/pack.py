"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.

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
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.dynamic.pack_tik import pack_tik
from impl.pack import op_select_format as pack_op_select_format


def op_select_format(x, y, axis, kernel_name="pack"):
    """
    op_select_format
    """
    return pack_op_select_format(x, y, axis, kernel_name)


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("Pack", op_mode="dynamic", support_fusion=False)
def pack_compute(x, y, axis, kernel_name="pack"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.

    Parameters
    ----------
    x : A list of `dict`.dict include keys shape and dtype
    y: dict of output_data, dict include keys shape and dtype
    axis : int, in the range [-rank(values)-1, rank(values)]
    kernel_name : cce kernel name, default value is "pack"

    Returns
    -------
    res : placeholder and res
    """
    res = tbe.concat(x, axis)

    return res


def convert_pack_2_concat_input(x, y, axis):
    """
    convert_pack_2_concat_input

    when pack without last dim, input axis change to axis + 1
    when pack with last dim, input shape change to [input , 1]
    """
    if axis is None:
        return x, y, axis

    if axis < -1:
        return x, y, axis + 1

    for item in x:
        if is_unknown_rank_input(item):
            continue
        ori_shape = item.get("ori_shape")
        shape = item.get("shape")
        if len(shape) == len(ori_shape) and axis in (-1, len(shape)):
            item["shape"] = list(item["shape"])
            item["ori_shape"] = list(item["ori_shape"])
            item["shape"].append(1)
            item["ori_shape"].append(1)
            item["range"] = list(item["range"])
            item["range"].append((1, 1))

    return x, y, axis


# 'pylint: disable=too-many-locals
def pack_dsl(x, y, axis, kernel_name="pack"):
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
    dtype_x = x[0].get("dtype")

    concat_x, concat_y, concat_axis = convert_pack_2_concat_input(x, y, axis)
    # transfer concat_axis to tiling for dynamic shape
    tbe_context.get_context().add_compile_info("concat_dim", concat_axis)
    tbe_context.get_context().add_compile_info("attr_name", "axis")
    tbe_context.get_context().add_compile_info("attr_idx", 0)

    extra_params = {"axis": concat_axis, "same_input": True}
    ins = classify([concat_x], "concat", extra_params)
    schedules, tensors = [], []
    for (input_x_, axis_) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([input_x_], "concat")
            input_tensors = []
            for index, shape in enumerate(shape_x):
                data = tvm.placeholder(shape, dtype=dtype_x, name=f"data_{index}")
                input_tensors.append(data)
            res = pack_compute(input_tensors, concat_y, axis_, kernel_name)

            tensors.append([*input_tensors, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


@register_operator("Pack")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def pack(x, y, axis, kernel_name="pack"):
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
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        pack_dsl(x, y, axis, kernel_name)
    else:
        pack_tik(x, y, axis, kernel_name)
