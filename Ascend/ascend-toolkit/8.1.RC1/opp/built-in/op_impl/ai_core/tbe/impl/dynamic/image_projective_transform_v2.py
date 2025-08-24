"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

image_projective_transform_v2
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.dynamic.image_projective_transform import ImageProjectiveTransform


# 'pylint: disable=unused-argument
@register_operator("ImageProjectiveTransformV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def image_projective_transform_v2(images,
                               transforms,
                               output_shape,
                               fill_value,
                               transformed_image,
                               interpolation,
                               fill_mode="CONSTANT",
                               kernel_name="image_projective_transform_v2"):
    """
    Generate arg_min operator use arg_min

    Parameters
    ----------
    images: dict
        data of input, support "float16", "float32", "uint8", "int32".
    transforms: dict
        3 x 3 projective transformation matrix, support "float32".
    output_shape: dict
        shape of output, support "int32".
    interpolation: str
        interpolation method, support "NEAREST" or "BILINEAR".
    fill_mode: str
        An optional string, Default is "CONSTANT", support "REFLECT", "WRAP", "NEAREST" or "CONSTANT".
    y: dict
        index of output.
    kernel_name: str
        kernel name, default value is "image_projective_transform_v2"

    Returns
    -------
    tik_instance
    """
    images_dtype = images.get("dtype")
    transforms_dtype = transforms.get("dtype")
    output_shape_dtype = output_shape.get("dtype")

    # check input shape, format and dtype
    para_check.check_dtype(images_dtype, ("float16", "float32", "uint8", "int32"), param_name="images")
    para_check.check_dtype(transforms_dtype, ("float32",), param_name="transforms")
    para_check.check_dtype(output_shape_dtype, ("int32",), param_name="output_shape")

    obj = ImageProjectiveTransform(images_dtype, transforms_dtype, interpolation, fill_mode, kernel_name)
    if fill_value is not None:
        fill_value_dtype = fill_value.get("dtype")
        para_check.check_dtype(fill_value_dtype, ("float16", "float32", "uint8", "int32"), param_name="fill_value")
        obj.exist_fill_value_n = True

    return obj.img_compute()
