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
dynamic floordiv
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,redefined-argument-from-local
@register_operator_compute("FloorDiv", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def floor_div_compute(input_x, input_y, output_z, kernel_name='floor_div'):
    """
       floordiv compute
       calculating data's floordiv, res =floor(x / y)

       Parameters
       ----------
       input_x: TVM tensor
           the placeholder of input_x
       input_y: TVM tensor
           the placeholder of input_y
       output_z: dict
           dict of output
       kernel_name: str
           kernel name, default value is "floor_div"

       Returns
       -------
       res: TVM tensor
           the result of floordiv compute
    """
    dtype_x = input_x.dtype
    _, _, shape_broad = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                                    param_name_input1="input_x",
                                                    param_name_input2="input_y")

    if dtype_x == "int64":
        input_x = tbe.broadcast(input_x, shape_broad)
        input_y = tbe.broadcast(input_y, shape_broad)

        res = tbe.vdiv(input_x, input_y)
        return res

    if dtype_x != "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vdiv", "float32"):
        input_x = tbe.cast_to(input_x, 'float32')
        input_y = tbe.cast_to(input_y, 'float32')

        input_x = tbe.broadcast(input_x, shape_broad)
        input_y = tbe.broadcast(input_y, shape_broad)
    else:
        input_x = tbe.broadcast(input_x, shape_broad)
        input_y = tbe.broadcast(input_y, shape_broad)

    res = tbe.vdiv(input_x, input_y)

    if dtype_x != "float16" and tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310":
        res = tbe.cast_to(res, "float16")

    if tbe_platform.api_check_support("tbe.dsl.floor", "f322f32"):
        res = tbe.cast_to(res, "float32")
        res = tbe.floor(res, "float32")
    else:
        res = tbe.floor(res)

    res = tbe.cast_to(res, dtype_x)

    return res


@register_operator("FloorDiv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def floor_div(input_x, input_y, output_z, kernel_name="floor_div"):
    """
      algorithm: floordiv
      calculating data's floordiv, res =floor(x / y)

      Parameters
      ----------
      input_x: dict
      input_y: dict
      output_z: dict
      kernel_name: str, default value is "floor_div"

      Returns
      -------
      None
    """
    # check dtype of input_x/input_y
    input_dtype_x = input_x.get("dtype").lower()
    input_dtype_y = input_y.get("dtype").lower()
    check_list = ['int8', 'uint8', 'int32', 'float16', 'float32', "bfloat16"]
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "int64"):
        check_list.append("int64")

    para_check.check_dtype(input_dtype_x, check_list, param_name="input_x")
    para_check.check_dtype(input_dtype_y, check_list, param_name="input_y")
    if input_dtype_x != input_dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal("floor_div", "input_x", "input_y",
                                                                 str(input_dtype_x), str(input_dtype_y))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with tbe.compute():
            x_shape, y_shape = \
                shape_util.variable_shape([input_x, input_y])
            tensor_x = tvm.placeholder(x_shape, input_dtype_x, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, input_dtype_y, "tensor_y")
            res = floor_div_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
