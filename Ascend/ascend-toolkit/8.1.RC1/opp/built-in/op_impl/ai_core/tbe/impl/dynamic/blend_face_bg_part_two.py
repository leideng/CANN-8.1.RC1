# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
blend_face_bg_part_two
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl import constant_util as constant


def _check_dtype(acc_face_dtype, acc_mask_dtype, max_mask_dtype, bg_img_dtype):
    """check input dtype"""
    para_check.check_dtype(acc_face_dtype, (constant.DATA_TYPE_FP32,))
    para_check.check_dtype(acc_mask_dtype, (constant.DATA_TYPE_FP32,))
    para_check.check_dtype(max_mask_dtype, (constant.DATA_TYPE_FP32,))
    para_check.check_dtype(bg_img_dtype, (constant.DATA_TYPE_FP32, constant.DATA_TYPE_UINT8))


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument
@register_operator_compute("blend_face_bg_part_two", op_mode="dynamic")
def blend_face_bg_part_two_compute(acc_face, acc_mask, max_mask, bg_img, fused_img, epsilon,
                                   kernel_name="blend_face_bg_part_two"):
    """
    calculating output
    """

    # cast bg_img_data
    if bg_img.dtype == constant.DATA_TYPE_UINT8:
        bg_img = tbe.cast_to(bg_img, constant.DATA_TYPE_FP16)
        bg_img = tbe.cast_to(bg_img, constant.DATA_TYPE_FP32)

    # calculate fusion face
    tmp_acc_mask = tbe.vadds(acc_mask, epsilon)
    fusion_face = tbe.vdiv(acc_face, tmp_acc_mask)

    # calculate fusion face with background
    neg_max_mask = tbe.vmuls(max_mask, tvm.const(-1.0, dtype=constant.DATA_TYPE_FP32))
    tmp_max_mask = tbe.vadds(neg_max_mask, tvm.const(1.0, dtype=constant.DATA_TYPE_FP32))
    bg_img_without_mask = tbe.vmul(bg_img, tmp_max_mask)
    fusion_face_without_bg = tbe.vmul(fusion_face, max_mask)

    res = tbe.vadd(fusion_face_without_bg, bg_img_without_mask)

    return res


@register_operator("BlendFaceBgPartTwo")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def blend_face_bg_part_two(acc_face, acc_mask, max_mask, bg_img, fused_img,
                           epsilon=1e-12, kernel_name="blend_face_bg_part_two"):
    """
    BlendFaceBgPartTwo op

    Parameters
    ----------
    acc_face: dict
        the dict of input acc_face, shape is [H, W, 3]
    acc_mask: dict
        the dict of input acc_mask, shape is [H, W, 3]
    max_mask: dict
        the dict of input max_mask, shape is [H, W, 3]
    bg_img: dict
        the dict of input bg_img, shape is [H, W, 3]
    fused_img: dict
        the dict of output fused_img, shape is [H, W, 3]
    epsilon: const
        the scalar of attr
    kernel_name: str
        kernel name, default value is "blend_face_bg_part_two"

    Returns
    -------
    None
    """

    bg_img_dtype = bg_img.get("dtype").lower()
    acc_face_dtype = acc_face.get("dtype").lower()
    acc_mask_dtype = acc_mask.get("dtype").lower()
    max_mask_dtype = max_mask.get("dtype").lower()
    _check_dtype(acc_face_dtype, acc_mask_dtype, max_mask_dtype, bg_img_dtype)

    tensor_list = []
    ins = classify([acc_face, acc_mask, max_mask, bg_img], OpPatternMode.ELEWISE)

    for (input_acc_face, input_acc_mask, input_max_mask, input_bg_img) in ins:
        with tbe.compute():
            shape_acc_face, shape_acc_mask, shape_max_mask, shape_bg_img = \
                shape_util.variable_shape([input_acc_face, input_acc_mask, input_max_mask, input_bg_img])

            scalar = get_attr_by_cls(epsilon, OpAttr(0, "epsilon", "Float", constant.EPSLON), constant.DATA_TYPE_FP32)

            acc_face_data = tvm.placeholder(shape_acc_face, acc_face_dtype, name="acc_face_data")
            acc_mask_data = tvm.placeholder(shape_acc_mask, acc_mask_dtype, name="acc_mask_data")
            max_mask_data = tvm.placeholder(shape_max_mask, max_mask_dtype, name="max_mask_data")
            bg_img_data = tvm.placeholder(shape_bg_img, bg_img_dtype, name="bg_face_data")

            res = blend_face_bg_part_two_compute(acc_face_data, acc_mask_data, max_mask_data, bg_img_data,
                                                 fused_img, scalar, kernel_name)

            tensor_list.append([acc_face_data, acc_mask_data, max_mask_data, bg_img_data, res])

        # auto schedule
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedule, config)
