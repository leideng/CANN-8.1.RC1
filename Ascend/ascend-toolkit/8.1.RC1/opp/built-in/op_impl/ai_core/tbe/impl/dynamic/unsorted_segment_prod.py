# Copyright 2020 Huawei Technologies Co., Ltd
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
unsorted_segment_prod
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from .unsorted_segment_min import op_select_format as unsorted_segment_prod_op_select_format
from .unsorted_segment_min import check_supported as unsorted_segment_prod_check_supported


# 'pylint: disable=too-many-arguments
def check_supported(x,
                    segment_ids,
                    num_segments,
                    y,
                    kernel_name="unsorted_segment_prod",
                    impl_mode=OpImplMode.HIGH_PRECISION):
    return unsorted_segment_prod_check_supported(x, segment_ids, num_segments, y, kernel_name, impl_mode)


def op_select_format(x, segment_ids, num_segments, y,
                     kernel_name="unsorted_segment_prod",
                     impl_mode=OpImplMode.HIGH_PRECISION):
    return unsorted_segment_prod_op_select_format(x, segment_ids, num_segments, y, kernel_name, impl_mode)


def unsorted_segment_prod_compute(x, segment_ids, var_num_segments, y,
                                 kernel_name="segmentensor_prod", impl_mode=OpImplMode.HIGH_PRECISION):
    check_ids = False
    if impl_mode == OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX:
        check_ids = True
    res = tbe.segment(x, segment_ids, var_num_segments, 1, "segmentensor_prod", check_ids)
    return res


@register_operator("UnsortedSegmentProd")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def unsorted_segment_prod(x,
                          segment_ids,
                          num_segments,
                          y_dict,
                          kernel_name="UnsortedSegmentProd",
                          impl_mode=OpImplMode.HIGH_PRECISION):
    """
    unsorted_segment_sum entry interface

    Parameters
    ----------
    x: input params shape, dtype and range
    segment_ids: segment_ids shape, dtype and range
    num_segments: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegmentProd op

    Returns
    -------
    compile info
    """
    check_op_impl_mode(impl_mode, 
                       [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX],
                       kernel_name)
    x_dtype = x.get("dtype").lower()
    x_dtype_check_list = ("float32", "float16", "int32")
    para_check.check_dtype(x_dtype, x_dtype_check_list, param_name="x")

    segment_ids_dtype = segment_ids.get("dtype").lower()
    segment_ids_dtype_check_list = ("int32", "int64")
    para_check.check_dtype(segment_ids_dtype,
                           segment_ids_dtype_check_list,
                           param_name="segment_ids")

    num_segments_dtype = num_segments.get("dtype").lower()
    num_segments_dtype_check_list = ("int32", "int64")
    para_check.check_dtype(num_segments_dtype,
                           num_segments_dtype_check_list,
                           param_name="num_segments")

    y_dtype = y_dict.get("dtype").lower()
    para_check.check_dtype(y_dtype, x_dtype_check_list, param_name="y_dict")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, "x", "y", x_dtype, y_dtype)

    ins = classify([x, segment_ids, num_segments], OpPatternMode.SEGMENT, {"impl_mode": impl_mode})
    schedules, tensors = [], []
    for (input1, input2, input3) in ins:
        with tbe.compute():
            shape_x1, shape_x2, var_segments = \
                shape_util.variable_shape([input1, input2, input3], op_mode="segment")
            x_tensor = tvm.placeholder(shape_x1, name="var", dtype=x_dtype)
            ids_tensor = tvm.placeholder(shape_x2, name="segment_ids", dtype=segment_ids_dtype)
            segments_tensor = tvm.placeholder([1], name="num_segments", dtype=num_segments_dtype)
            res = unsorted_segment_prod_compute(x_tensor, ids_tensor, var_segments, y_dict, kernel_name, impl_mode)
            tensors.append([x_tensor, ids_tensor, segments_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
