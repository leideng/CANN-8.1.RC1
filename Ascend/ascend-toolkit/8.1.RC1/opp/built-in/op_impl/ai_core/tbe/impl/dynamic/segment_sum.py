# Copyright 2021 Huawei Technologies Co., Ltd
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
segment_sum
"""
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable=unused-argument,too-many-arguments
def check_supported(x, segment_ids, y, kernel_name="unsorted_segment_sum",
                    impl_mode=OpImplMode.HIGH_PRECISION):
    """
    dynamic -1 support
    segment_ids int64 not support
    static shape x_shape ends with 1 or lens equals 1 not support
    temporary support x_dtype of "float32" in compilestatic process
    """
    id_dtype = segment_ids.get("dtype").lower()
    x_dtype = x.get("dtype").lower()

    if id_dtype not in ("int32", "int64"):
        reason = "the segment_ids's dytpe not equeal int32 or int64, segment_ids_dtype=%s" % id_dtype
        return False, reason
    if x_dtype not in ("float32", "float16", "int32"):
        reason = "x_dtype not support, x_dtype=%s" % x_dtype
        return False, reason
    return True, ""


def op_select_format(x, segment_ids, y, kernel_name="unsorted_segment_sum",
                     impl_mode=OpImplMode.HIGH_PRECISION):
    """
    select format dynamically
    """
    input0_dtype = "float16,int32,float,float16,int32,float"
    input0_format = "ND,ND,ND,ND,ND,ND"
    input1_dtype = "int32,int32,int32,int64,int64,int64"
    input1_format = "ND,ND,ND,ND,ND,ND"
    input0 = gen_param(classify="input0", name="x",
                       datatype=input0_dtype,
                       format=input0_format,
                       unknownshape_format=input0_format)
    input1 = gen_param(classify="input1", name="segment_ids",
                       datatype=input1_dtype,
                       format=input1_format,
                       unknownshape_format=input1_format)
    output0 = gen_param(classify="output0", name="y",
                        datatype=input0_dtype,
                        format=input0_format,
                        unknownshape_format=input0_format)

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-arguments
def segment_sum_compute(x, segment_ids, var_num_segments, y, kernel_name="SegmentSum",
                        impl_mode=OpImplMode.HIGH_PRECISION):
    check_ids = False
    if impl_mode == OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX:
        check_ids = True
    res = tbe.segment(x, segment_ids, var_num_segments, 0, "segmentensor_sum", check_ids)
    return res


# 'pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-branches
# 'pylint: disable=superfluous-parens
@register_operator("SegmentSum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def segment_sum(x, segment_ids, y, kernel_name="segment_sum", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    Updates specified rows with values in v

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor x
    segment_id: dict
    y : dict
        shape and dtype of output tensor
    kernel_name : str
        kernel name, default value is "segment_sum"

    Returns
    -------
    tik_instance
    """
    check_op_impl_mode(impl_mode,
                       [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX],
                       kernel_name)
    x_dtype = x.get("dtype").lower()
    x_dtype_check_list = ("float32", "float16", "int32")
    para_check.check_dtype(x_dtype, x_dtype_check_list, param_name="x")

    segment_ids_dtype = segment_ids.get("dtype").lower()
    segment_ids_dtype_check_list = ("int32", "int64")
    para_check.check_dtype(segment_ids_dtype, segment_ids_dtype_check_list, param_name="segment_ids")

    y_dtype = y.get("dtype").lower()
    para_check.check_dtype(y_dtype, x_dtype_check_list, param_name="y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x", "y", x_dtype, y_dtype)
    num_segments_input = {"shape":(1,), "dtype":"int32", "ori_shape":(1,), "range":[[1, 1]]}
    if "const_value" in segment_ids:
        id_value = segment_ids.get("const_value")
        if isinstance(id_value, int):
            num_segments = id_value + 1
        else:
            num_segments = max(id_value) + 1
        num_segments_input.update({"const_value":num_segments})
    ins = classify([x, segment_ids, num_segments_input], OpPatternMode.SEGMENT,
            {"impl_mode": impl_mode, "is_segment": True})
    schedules, tensors = [], []
    for (input1, input2, input3) in ins:
        with tbe.compute():
            shape_x1, shape_x2, var_segments = \
                shape_util.variable_shape([input1, input2, input3], op_mode="segment")
            x_tensor = tvm.placeholder(shape_x1, name="var", dtype=x_dtype)
            segment_ids_tensor = tvm.placeholder(shape_x2, name="segment_ids", dtype=segment_ids_dtype)
            segments_tensor = tvm.placeholder([1], name="num_segments", dtype="int32")
            res = segment_sum_compute(x_tensor, segment_ids_tensor, var_segments, y, kernel_name, impl_mode)
            tensors.append([x_tensor, segment_ids_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
