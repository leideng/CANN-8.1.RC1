# Copyright 2019 Huawei Technologies Co., Ltd
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
cast
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import vand
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_common


def _new_alloc(ir_builder, dtype, shape, name, scope):
    """
    alloc memory for decl new buffer
    """
    buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)
    return new_buffer


def _kernel_ir(dst, src, dst_type, src_type):
    """
    convert a scale from src type to dst type
    NOTICE: SCALE ONLY
    """
    ir_builder = tvm.tir.ir_builder.create()
    in_tensor = src[0]
    a_ub = _new_alloc(ir_builder,
                      src_type,
                      in_tensor.shape,
                      "a_ub",
                      scope=tbe_platform.scope_ubuf)
    out_tensor = dst[0]
    b_ub = _new_alloc(ir_builder,
                      dst_type,
                      in_tensor.shape,
                      "b_ub",
                      scope=tbe_platform.scope_ubuf)

    reg = ir_builder.allocate(dst_type, (1, ), name='reg', scope=tbe_platform.scope_reg)
    ir_builder.emit(
        tvm.call_extern(src_type, "copy_gm_to_ubuf", a_ub.access_ptr("w"),
                        in_tensor.access_ptr("r"), 0, 1, 1, 0, 0))
    ir_builder.emit(
        tvm.call_extern(src_type, "reg_mov",
                        tvm.call_extern(dst_type, "reg", reg[0]),
                        a_ub.access_ptr('r', offset=0)))
    ir_builder.emit(
        tvm.call_extern(dst_type, "reg_mov", b_ub.access_ptr('w', offset=0),
                        tvm.call_extern(dst_type, "reg", reg[0])))
    ir_builder.emit(
        tvm.call_extern(dst_type,
                        "copy_ubuf_to_gm", out_tensor.access_ptr('w'),
                        b_ub.access_ptr("r"), 0, 1, 1, 0, 0))

    return ir_builder.get()


# 'pylint: disable=inconsistent-return-statements
def _int8_uint8_process(data, dst_type):
    """
    deal with src dtype=int8 and uint8 case
    """
    if dst_type == "float16":
        return tbe.cast_to(data, "float16")

    if dst_type == "float32":
        data_fp16 = tbe.cast_to(data, "float16")
        return tbe.cast_to(data_fp16, "float32")

    if dst_type == "int32":
        data_fp16 = tbe.cast_to(data, "float16")
        return tbe.cast_to(data_fp16, "int32")

    if dst_type == "uint8":
        data_fp16 = tbe.cast_to(data, "float16")
        abs_fp16 = tbe.vabs(data_fp16)
        return tbe.cast_to(abs_fp16, "uint8")

    error_manager_vector.raise_err_specific_reson("cast", "The cast_cce_aicore only support int8/uint8"
                                                  "cast to float16,float32,int32,uint8.")


# 'pylint: disable=inconsistent-return-statements
def _int32_process(data, dst_type):
    """
    deal with src dtype=int32 case
    """
    if dst_type == "bool":
        const_one = tvm.const(1.0, "float16")
        shape_data = shape_util.shape_to_list(data.shape)
        const_broad = tbe.broadcast(const_one, shape_data)

        data = tbe.cast_to(data, "float16", True)
        x_abs = tbe.vabs(data)
        x_min = tbe.vmin(x_abs, const_broad)
        y_abs = tbe.vabs(x_min)
        return tbe.cast_to(y_abs, "int8", True)

    if dst_type == "int8":
        const_ff = tvm.const(255, "int32")
        shape_data = shape_util.shape_to_list(data.shape)
        const_broad = tbe.broadcast(const_ff, shape_data)
        data_and = vand(data, const_broad)

        data_fp16 = tbe.cast_to(data_and, "float16")
        res = util_common.uint8_int8_overflow_proc(data_fp16, "int8")
        return res

    if dst_type == "uint8":
        const_ff = tvm.const(255, "int32")
        shape_data = shape_util.shape_to_list(data.shape)
        const_broad = tbe.broadcast(const_ff, shape_data)
        data_and = vand(data, const_broad)

        data_fp16 = tbe.cast_to(data_and, "float16")
        tensor_0 = tbe.vmuls(data_fp16, 0)
        tensor_256 = tbe.vadds(tensor_0, 256)
        result = tbe.vmod(data_fp16, tensor_256)
        result = tbe.cast_to(result, "float16")
        return tbe.cast_to(result, "uint8", True)

    if dst_type == "float32":
        return tbe.cast_to(data, "float32")

    if dst_type == "float16":
        return tbe.cast_to(data, "float16")

    error_manager_vector.raise_err_specific_reson("cast", "The cast_cce_aicore only support int32"
                                                  "cast to bool,int8,uint8,float32,float16.")


# 'pylint: disable=inconsistent-return-statements
def _float32_process(data, dst_type):
    """
    deal with src dtype=float32 case
    """
    if dst_type == "int32":
        return tbe.cast_to(data, "int32")

    if dst_type == "float16":
        return tbe.cast_to(data, "float16")

    error_manager_vector.raise_err_specific_reson("cast", "The cast_cce_aicore only support float32"
                                                  "cast to int32,float16.")


# 'pylint: disable=inconsistent-return-statements
def _float16_process(data, dst_type):
    """
    deal with src dtype=float16 case
    """
    if dst_type == "float16":
        return data

    if dst_type == "float32":
        return tbe.cast_to(data, "float32")

    if dst_type == "int32":
        return tbe.cast_to(data, "int32")

    if dst_type == "int8":
        data_int32 = tbe.cast_to(data, "int32")
        const_ff = tvm.const(255, "int32")
        shape_data = shape_util.shape_to_list(data.shape)
        const_broad = tbe.broadcast(const_ff, shape_data)
        data_and = vand(data_int32, const_broad)

        data_fp16 = tbe.cast_to(data_and, "float16")
        res = util_common.uint8_int8_overflow_proc(data_fp16, "int8")
        return res
    
    if dst_type == "bool":
        res = tbe.vcmp(data, 0.0, "ne")
        return res

    if dst_type == "uint8":
        if not tbe_platform.api_check_support("tbe.dsl.cast_to", "s322f16") and \
                tbe_platform.api_check_support("tbe.dsl.vmod", "float16"):
            return tbe.cast_to(data, "uint8", True)
        data_int32 = tbe.cast_to(data, "int32")
        data_fp16 = tbe.cast_to(data_int32, "float16")
        tensor_0 = tbe.vmuls(data_fp16, 0)
        tensor_256 = tbe.vadds(tensor_0, 256)
        result = tbe.vmod(data_fp16, tensor_256)
        result = tbe.cast_to(result, "float16")
        return tbe.cast_to(result, "uint8", True)

    error_manager_vector.raise_err_specific_reson("cast", "The cast_cce_aicore only support float16"
                                                  "cast to float32,int32,uint8,int8,bool.")


def _cast_dsttype_conversion(dst_type):
    if dst_type == 0:
        dst_type = "float32"
    if dst_type == 1:
        dst_type = "float16"
    if dst_type == 2:
        dst_type = "int8"
    if dst_type == 3:
        dst_type = "int32"
    if dst_type == 4:
        dst_type = "uint8"
    if dst_type == 10:
        dst_type = "uint64"
    if dst_type == 12:
        dst_type = "bool"
    return dst_type


# 'pylint: disable=unused-argument
def check_supported(input_x, output_y, dst_type, kernel_name="cast"):
    """
    verify the types of cast supported by tbe
    """
    src_type = input_x.get("dtype").lower()

    check_result = False
    if src_type == "bool":
        src_type = "int8"

    dst_type = _cast_dsttype_conversion(dst_type)

    check_list = []
    if src_type == "float16":
        check_list = ["float16", "float32", "int32", "uint8", "int8", "bool"]
    elif src_type == "float32":
        check_list = ["float16", "int32"]
    elif src_type == "int8":
        check_list = ["float32", "float16", "int32", "uint8"]
    elif src_type == "uint8":
        check_list = ["float32", "float16", "int32"]
    elif src_type == "int32":
        check_list = ["bool", "uint8", "int8", "float32", "float16"]

    src_shape = input_x.get("ori_shape")

    if (len(src_shape) == 1 and src_shape[0] == 1) or len(src_shape) == 0:
        if src_type == "int64":
            check_list = ["int32", "float32"]

    if dst_type in check_list:
        check_result = True

    if check_result:
        return True, ""
    reason = "when input dtyps is %s, output dtype[%s] is not in checklist:%s" % (src_type, dst_type, str(check_list))
    return False, reason


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,inconsistent-return-statements
@register_operator_compute("cast", op_mode="static", support_fusion=True)
def cast_compute(data, output_y, dst_type, kernel_name="cast"):
    """
    core func of tensor casting. cast a tensor form src data type to dst data
    type. restrictions of input algorithms are as follow
    only types' groups blow are support tensor process:
        float16->float32
        float16->int32
        float32->float16
        float32->int32
        int8->float32
        uint8->float32
        int8->float16
        uint8->float16
        int8->int32
        uint8->int32
        int32->uint8 // number out of [0,255] can get unexpected result
        int32->int8 // number out of [-128,127] can get unexpected result
        int32->float32 // For tans with fp16, only guarantees
                        number in [-1023,1023] get correct result
        int32->float16 // only guarantees
                        number in [-1023,1023] get correct result
    Parameters
    ----------
    placeholders: list.
        the input tensor
    src_type: str
        the input data type.
    dst_type: str
        the output data type.

    Returns
    -------
        the compute result tensor with type dst_type
    """
    src_data_type = data.dtype
    para_check.check_dtype(src_data_type, ("float16", "float32", "int8", "uint8", "int32"), param_name="input_x")

    dst_type = _cast_dsttype_conversion(dst_type)

    if src_data_type in ("int8", "uint8"):
        return _int8_uint8_process(data, dst_type)

    if src_data_type == "float32":
        return _float32_process(data, dst_type)

    if src_data_type == "float16":
        return _float16_process(data, dst_type)

    if src_data_type == "int32":
        return _int32_process(data, dst_type)

    error_manager_vector.raise_err_specific_reson("cast", "The cast_cce_aicore don't support this situation")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def cast(input_x, output_y, dst_type, kernel_name="cast"):
    """
    cast a tensor/scaler with input shape form src data type to dst data
    type. restrictions of input algorithms are as follow
    only types' groups blow are support tensor process:
        float16->float32
        float16->int32
        float32->float16
        float32->int32
        int8->float32
        uint8->float32
        int8->float16
        uint8->float16
        int8->int32
        uint8->int32
        int32->uint8 // number out of [0,255] can get unexpected result
        int32->int8 // number out of [-128,127] can get unexpected result
        int32->float32 // For tans with fp16, only guarantees
                        number in [-1023,1023] get correct result
        int32->float16 // only guarantees
                        number in [-1023,1023] get correct result
    scale convert support:(means only support shape [1,])
        int64->int32
        int64->float32

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape as input,
        and the dtype is the dst dtype need to cast
    kernel_name : str
        cce kernel name, default value is cast

    Returns
    -------
    None
    """
    shape = shape_util.scalar2tensor_one(input_x.get("shape"))
    src_type = input_x.get("dtype").lower()
    para_check.check_shape(shape, param_name="input_x")

    if src_type == "bool":
        src_type = "int8"

    dst_type = _cast_dsttype_conversion(dst_type)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=src_type)
    if src_type == "int64":
        para_check.check_dtype(dst_type, ("float32", "int32"), param_name="dst_type")
        res = tvm.extern(
            [fuseshape], [data],
            lambda ins, outs: _kernel_ir(outs, ins, dst_type, "int64"),
            name="res",
            dtype=dst_type)
        tensor_list = [data, res]
        schedule = tvm.create_schedule(res.op)
        with tbe_build.build_config():
            tvm.build(schedule, tensor_list, "cce", name=kernel_name)
    else:
        with tvm.target.cce():
            res = cast_compute(data, output_y, dst_type, kernel_name)
            sch = auto_schedule(res)
        config = {
            "print_ir": False,
            "name": kernel_name,
            "tensor_list": [data, res],
            "bool_storage_as_1bit": False
        }
        build(sch, config)
