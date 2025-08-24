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

"""CusCorrectionMul op"""
import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from mindspore.ops.op_info_register import op_info_register
from mindspore.ops.op_info_register import TBERegOp
from mindspore.ops.op_info_register import DataType

SHAPE_SIZE_LIMIT = 2147483648

cus_correction_mul_op_info = TBERegOp("CusCorrectionMul") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("cus_correction_mul.so") \
    .compute_cost(10) \
    .kernel_name("cus_correction_mul") \
    .partial_flag(True) \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "x", None, "required", None) \
    .input(1, "batch_std", None, "required", None) \
    .input(2, "running_std", None, "required", None) \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(cus_correction_mul_op_info)
def _correction_mul_tbe():
    """CusCorrectionMul TBE register"""
    return


@register_op_compute("CusCorrectionMul", op_mode="static", support_fusion=True)
def correction_mul_compute(x, batch_std, running_std, kernel_name="cus_correction_mul"):
    """CusCorrectionMul compute"""
    shape_x = shape_util.shape_to_list(x.shape)
    factor = tbe.vdiv(batch_std, running_std)
    factor_b = tbe.broadcast(factor, shape_x)
    res = tbe.vmul(x, factor_b)
    return res


@para_check.check_input_type(dict, dict, dict, dict, int, str)
def cus_correction_mul(x, batch_std, running_std, y, channel, kernel_name="cus_correction_mul"):
    """CorrectionMul op"""
    shape = x.get("shape")
    data_format = x.get("format")
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape)
    para_check.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    inp_dtype = x.get("dtype").lower()
    if not inp_dtype in ["float16", "float32"]:
        raise RuntimeError("Dtype of input only support float16, float32")

    x_t = tvm.placeholder(shape, name="x", dtype=inp_dtype)
    shape_c = [1] * len(shape)
    shape_c[channel] = batch_std.get("ori_shape")[0]
    if data_format == "NC1HWC0" and channel == 1:
        shape_c = batch_std.get("shape")
    batch_std_t = tvm.placeholder(shape_c, name="batch_std", dtype=inp_dtype)
    running_std_t = tvm.placeholder(shape_c, name="running_std", dtype=inp_dtype)
    res = correction_mul_compute(x_t, batch_std_t, running_std_t, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [x_t, batch_std_t, running_std_t, res]}

    tbe.build(sch, config)
