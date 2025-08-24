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
bn_training_reduce
"""
import math

from tbe.common.platform import CceProductParams
from impl.util.platform_adapter import build_config
import te.platform as tbe_platform
import te.lang.cce as tbe
from tbe.dsl.instrinsic import cce_util
from tbe import tvm
from impl.util.util_common import gen_range
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from tbe.dsl.instrinsic import cce_emitinsn_params
from impl.dynamic.bn_training_reduce import op_select_format as bn_op_select_format
from impl.dynamic.bn_training_reduce import get_op_support_info as bn_get_op_support_info
from impl.dynamic.bn_training_reduce import bn_training_reduce as bn_training_reduce_dynamic


# 'pylint: disable = unused-argument
# 'pylint: disable=invalid-name,redefined-builtin,too-many-statements
def check_supported(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    check supported
    """
    return True, ""


def get_op_support_info(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    get_op_support_info
    """
    return bn_get_op_support_info(x, sum, square_sum, kernel_name)


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin, too-many-locals
def op_select_format(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    1. when input(x)'s ori_shape is [1, ? ,1, ?] and the format is NCHW,
    the Op BNTrainingReduce can support NCHW.
    > for example:
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingReduce can process with NC1HWC0:
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    2. In other scenes, the Op BNTrainingReduce can support NC1HWC0 and NDC1HWC0
    > for example:
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    return bn_op_select_format(x, sum, square_sum, kernel_name)


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW", "NDC1HWC0"):
        error_manager_vector.raise_err_specific_reson("bn_training_reduce",
                                                      "The data format only supports NC1HWC0, NDC1HWC0 and NCHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_manager_vector.raise_err_specific_reson("bn_training_reduce",
                                                          "The origin format only supports NCHW when format is NCHW")


def _reduce_compute_5hd(x):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    square_x = tbe.vmul(x, x)

    axis = [0, 2, 3]
    sum_x, square_sum_x = tbe.tuple_sum([x, square_x], axis, True)

    res = [sum_x, square_sum_x]

    return res


def _reduce_compute_nd(x, sum):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    origin_format = sum.get("ori_format")
    shape = shape_util.shape_to_list(x.shape)
    axis = list(range(len(shape)))

    if origin_format == "NCHW":
        axis.pop(1)

    for _, i in enumerate(range(len(shape))):
        if shape[i] == 1 and i in axis:
            axis.remove(i)

    square_x = tbe.vmul(x, x)
    sum_x = tbe.sum(x, axis, False)
    square_sum_x = tbe.sum(square_x, axis, False)

    # Output has been reversed because of binary_reduce_output_reversed
    res = [square_sum_x, sum_x]

    return res


# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
@tvm._ffi.register_func("tvm.intrin.cce.bn_reduce_sum")
def bn_reduce_sum(stmt_op):
    """
    Collapse second input tensor to one repeat
    and use vcadd to calculate sum to output
    Schedule for ND
    Including definition for several operator specific intrinsic instructions
    bn_reduce_sum
    """
    # Get input and output buffers
    input_size_list = [1]
    for_extents = []
    ir_builder = tvm.tir.ir_builder.create()
    cce_util.get_init_op(stmt_op)

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.tir.For):
            input_size_list[0] = input_size_list[0] * _stmt.extent.value
            for_extents.append(_stmt.extent.value)

    tvm.tir.stmt_functor.ir_transform(stmt_op, None, _post_order_for, ["tir.For"])
    ins, outs = cce_util.get_buffer(stmt_op, need_unique=True, need_origin_adress=True)
    in_buffer = ins[1]
    out_buffer = outs[0]
    input_size = input_size_list[0]

    # Check if input can be collapsed into one repeat
    vector_inst_one_repeat_size = \
        tbe_platform.cce_params.VECTOR_INST_BLOCK_WIDTH // cce_util.get_align_factor(in_buffer.dtype)[1]


    # get reduce_axis shape
    if len(for_extents) == 1:
        input_reduce_axis_shape = for_extents[0]
        ub_loop_num = 1
    else:
        input_reduce_axis_shape = for_extents[0]
        ub_loop_num = for_extents[1]

    collapse_loop_num = \
        math.log(input_reduce_axis_shape / vector_inst_one_repeat_size, 2)

    # judge reduce_shape is remaining or not after dichotomy add
    remain_flag = False
    collapse_repeat = 0
    if not collapse_loop_num.is_integer():
        collapse_repeat = int(math.pow(2, int(collapse_loop_num)))
        out_of_collapse_repeat = \
            input_reduce_axis_shape / vector_inst_one_repeat_size - \
            collapse_repeat
        if not out_of_collapse_repeat.is_integer():
            error_detail = "Input size is not aligned:%s" % input_reduce_axis_shape
            error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)
        remain_flag = True

    # Do Emit Insn
    def collapse(ir_b, buffer, current_size):
        """Function to do emit insn"""
        repeat = current_size // 2 / vector_inst_one_repeat_size
        tail_flag = False
        if not repeat.is_integer():
            tail_flag = True
        repeat = int(repeat)

        ir_b.emit(tvm.call_extern(
            buffer.dtype,
            "vadd",
            buffer.access_ptr("rw", offset=0),
            buffer.access_ptr("r", offset=0),
            buffer.access_ptr("r", offset=8),
            repeat, 1, 2, 2, 8, 16, 16))

        # solve tail vadd
        if tail_flag:
            tail_mask = \
                (current_size - repeat * 2 * vector_inst_one_repeat_size) // 2
            tbe_platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype, tail_mask)
            ir_b.emit(tvm.call_extern(
                buffer.dtype,
                "vadd",
                buffer.access_ptr("rw",
                                  offset=repeat*vector_inst_one_repeat_size),
                buffer.access_ptr("r",
                                  offset=repeat*2*vector_inst_one_repeat_size),
                buffer.access_ptr("r",
                                  offset=repeat*2*vector_inst_one_repeat_size +
                                  8),
                1, 1, 2, 2, 0, 0, 0))
            tbe_platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype)
        return current_size // 2

    # emit vadd
    cur_size = input_size
    for loop in range(int(collapse_loop_num)):
        cur_size = collapse(ir_builder, in_buffer, cur_size)

    if remain_flag:
        # solve remain repeat
        mask_bits = \
            input_reduce_axis_shape / collapse_repeat - \
            vector_inst_one_repeat_size
        add_repeat_stride = int(8 + mask_bits / 8)
        tbe_platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype, mask_bits)
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vadd",
            in_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0),
            in_buffer.access_ptr("r", offset=vector_inst_one_repeat_size),
            ub_loop_num, 1, 1, 1,
            add_repeat_stride,
            add_repeat_stride,
            add_repeat_stride))

        # emit vcadd for remain
        tbe_platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype)
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vcadd",
            out_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0),
            ub_loop_num, 1, 1, add_repeat_stride))
    else:
        # emit vcadd for no remain
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vcadd",
            out_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0), ub_loop_num, 1, 1, 8))

    return ir_builder.get()


# 'pylint: disable=locally-disabled,too-many-locals
@tvm._ffi.register_func("tvm.intrin.cce.binary_reduce_output_reversed")
def binary_reduce_output(stmt_op):
    """Move reduce results to two destinations"""
    # Get input and output buffers
    input_size_list = [1]
    ir_builder = tvm.tir.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.tir.For):
            input_size_list[0] = input_size_list[0] * _stmt.extent.value

    def new_alloc(tvm_ib, dtype, shape, name, scope):
        """Funtion to alloc mem"""
        buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
        return new_buffer
    _ = tvm.tir.stmt_functor.ir_transform(stmt_op, None, _post_order_for, ["tir.For"])
    ins, outs = cce_util.get_buffer(stmt_op)
    # Alloc second buffer for binary collection
    out_buffer_sec = cce_emitinsn_params.cceEmitParamsIns.get_param("binary_reduce"
                                                                    "_output_buffer")
    in_buffer = ins[0], ins[1]
    out_buffer = outs[0], out_buffer_sec
    input_size = input_size_list[0]
    output_size = input_size
    block_unit = cce_util.get_align_factor(in_buffer[0].dtype)[0]
    remain_buffer = new_alloc(ir_builder, out_buffer[0].dtype, (block_unit,),
                              "copy_part_0", tbe_platform.cce_params.scope_ubuf)
    remain_buffer_sec = new_alloc(ir_builder, out_buffer[1].dtype,
                                  (block_unit,), "copy_part_1",
                                  tbe_platform.cce_params.scope_ubuf)
    burst_len = max(output_size // block_unit, 1)
    remains = max(output_size - burst_len * block_unit, 0)
    remains_fill = block_unit - remains

    # Main part
    global_offset = out_buffer[0].elem_offset
    ir_builder.emit(
        tvm.call_extern(out_buffer[0].dtype, "copy_ubuf_to_gm",
                        out_buffer[0].access_ptr("rw"),
                        in_buffer[1].access_ptr("r"),
                        0,
                        1,
                        burst_len, 0, 0))
    ir_builder.emit(
        tvm.call_extern(out_buffer[1].dtype, "copy_ubuf_to_gm",
                        out_buffer[1].access_ptr("rw", offset=global_offset),
                        in_buffer[0].access_ptr("r"),
                        0,
                        1,
                        burst_len, 0, 0))
    # Remain part
    if remains > 0:
        with ir_builder.for_range(0, block_unit, name="copy_part_fill_loop") \
                as reg_mov_loop:
            ir_builder.emit(tvm.call_extern(
                remain_buffer.dtype, "reg_mov",
                remain_buffer.access_ptr("rw", offset=reg_mov_loop),
                in_buffer[1].access_ptr("r",
                                        offset=burst_len * block_unit -
                                        remains_fill + reg_mov_loop)))
            ir_builder.emit(tvm.call_extern(
                remain_buffer_sec.dtype, "reg_mov",
                remain_buffer_sec.access_ptr("rw", offset=reg_mov_loop),
                in_buffer[0].access_ptr("r",
                                        offset=burst_len *
                                        block_unit -
                                        remains_fill + reg_mov_loop)))
        ir_builder.emit(
            tvm.call_extern(out_buffer[0].dtype,
                            "copy_ubuf_to_gm",
                            out_buffer[0].access_ptr("rw",
                                                     offset=burst_len *
                                                     block_unit - remains_fill),
                            remain_buffer.access_ptr("r"),
                            0,
                            1,
                            1, 0, 0))
        ir_builder.emit(
            tvm.call_extern(out_buffer[1].dtype,
                            "copy_ubuf_to_gm",
                            out_buffer[1].access_ptr("rw",
                                                     offset=global_offset +
                                                     burst_len * block_unit -
                                                     remains_fill),
                            remain_buffer_sec.access_ptr("r"),
                            0,
                            1,
                            1, 0, 0))
    return ir_builder.get()


# 'pylint: disable=locally-disabled,too-many-branches
def bn_training_reduce_schedule_nd(res, core_num=None):
    """bn_training_reduce schedule method"""
    cce_emitinsn_params.cceEmitParamsIns.clear_param()
    # Prepare extra tensors
    # Step 1: Get two output tensors
    # Step 2: Merge two output tensors into Dummy
    # Step 3: Move UB data to GM tensor
    output_first = res[0]  # Square Sum
    output_second = res[1]  # Sum
    final_output = tvm.compute(output_first.shape,
                               lambda *indices: output_first(*indices) + output_second(*indices),
                               name="DummyYummySweety")
    is_cast = False
    if "cast" in output_second.op.input_tensors[0].name:
        is_cast = True
    # Calculate block split factor by axis_n_size and core_num
    c_dim_index = 1
    if len(res[0].shape) < 2:
        c_dim_index = 0
    axis_n_size = int(res[0].shape[c_dim_index])
    if not core_num:
        core_num = int(tbe_platform.get_soc_spec("CORE_NUM"))
    # Multi core kernel requires aligned output
    element_size = cce_util.get_align_factor(output_first.dtype)[1]
    block_element_num = tbe_platform.cce_intrin_md.ALIGNMENT_BYTES // element_size
    estimate_block_split_factor = max(axis_n_size // core_num, 8)
    nearest_aligned_factor = estimate_block_split_factor % block_element_num
    # Decrease core_num for aligned output
    if estimate_block_split_factor < block_element_num and core_num > 1:
        return bn_training_reduce_schedule_nd(res, core_num - 1)
    # Round to the nearest
    block_split_factor = estimate_block_split_factor - nearest_aligned_factor
    # Calculate UB split
    ub_size = CceProductParams().getParams("Unified_Buffer") // 2
    reduce_data_num = 1
    reduce_data_factor = 2
    if is_cast:
        reduce_data_factor = 3
    for reduce_axis in output_first.op.reduce_axis:
        reduce_data_num *= int(reduce_axis.dom.extent)
    reduce_data_num *= reduce_data_factor
    max_possible_loop = ub_size // (element_size * reduce_data_num)
    actual_loop = 1
    for loop in range(max_possible_loop - 1, 0, -1):
        if block_split_factor % loop == 0:
            actual_loop = loop
            break
    # Force aligned if multi-core is enabled
    if actual_loop < block_element_num and actual_loop < block_split_factor and core_num > 1:
        actual_loop = block_element_num

    # Find all tensors
    if is_cast:
        # With Cast, prepare tensor parameters
        mul_tensor = output_first.op.input_tensors[0]
        cast_tensor = mul_tensor.op.input_tensors[0]
        res_input = cast_tensor.op.input_tensors[0]
        input_tensor_next = [cast_tensor]  # First compute tensor is cast_tensor
        ub_tensors = [cast_tensor, mul_tensor, output_first, output_second]
    else:
        # Without Cast, prepare tensor parameters
        cast_tensor = None
        mul_tensor = output_first.op.input_tensors[0]
        res_input = mul_tensor.op.input_tensors[0]
        input_tensor_next = [mul_tensor, output_second]  # First compute tensor is cast_tensor
        ub_tensors = [mul_tensor, output_first, output_second]

    # Create original schedule
    sch = tvm.create_schedule(final_output.op)

    #  DataFlow Control
    # Read input in
    input_tensor_ub = sch.cache_read(res_input, tbe_platform.cce_params.scope_ubuf, input_tensor_next)
    ub_tensors.append(input_tensor_ub)
    #  Compute procedure in ubuf
    for ub_tens in ub_tensors:
        sch[ub_tens].set_scope(tbe_platform.cce_params.scope_ubuf)

    #  Split axis Control
    outer, inner = \
        sch[final_output].split(sch[final_output].op.axis[c_dim_index],
                                factor=block_split_factor)
    ub_outer, ub_inner = sch[final_output].split(inner, factor=actual_loop)
    sch[final_output].bind(outer, tvm.thread_axis("blockIdx.x"))

    # Compute Control
    compute_at_axis = ub_outer
    for ub_tens in ub_tensors:
        sch[ub_tens].compute_at(sch[final_output], compute_at_axis)

    # EmitInsn
    def emit_on_self(tensor, axisnum=0, op="dma_copy"):
        """Do emit insn"""
        sch[tensor].emit_insn(sch[tensor].op.axis[axisnum], op)

    def emit_on_self_ex(tensor, axis, op="dma_copy"):
        """Do emit insn"""
        sch[tensor].emit_insn(axis, op)

    # Fake results
    emit_on_self(input_tensor_ub, 0)
    if is_cast:
        emit_on_self(cast_tensor, 0, cast_tensor.op.tag.split('|')[0])
    emit_on_self(mul_tensor, 0, mul_tensor.op.tag)

    sch[output_first].pragma(sch[output_first].op.axis[c_dim_index], "emit_insn", "bn_reduce_sum")
    sch[output_second].pragma(sch[output_second].op.axis[c_dim_index], "emit_insn", "bn_reduce_sum")
    sch[output_first].double_buffer()
    sch[output_second].double_buffer()

    emit_on_self_ex(final_output, ub_inner, "binary_reduce_output_reversed")

    def new_alloc(dtype, shape, name):
        """Alloc mem"""
        new_buffer = tvm.decl_buffer(shape, dtype, name=name, scope="", data=None)
        return new_buffer

    out_buffer_sec = new_alloc(final_output.dtype, (block_split_factor,), "reduce_sec_output_gm")
    cce_emitinsn_params.cceEmitParamsIns.insert_param("binary_reduce_output_buffer", out_buffer_sec)
    tensor_list = [res_input, final_output, out_buffer_sec]

    return sch, tensor_list


@tbe_platform.fusion_manager.fusion_manager.register("bn_training_reduce")
def bn_training_reduce_compute(x, sum, square_sum,
                               kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
    data_format = sum.get("format")
    if data_format in ("NC1HWC0", "NDC1HWC0"):
        res = _reduce_compute_5hd(x)
    else:
        res = _reduce_compute_nd(x, sum)
    return res



def bn_training_reduce_static(x, sum, square_sum,
                              kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    data_format = x.get("format")
    origin_format = x.get("ori_format")
    _check_format(data_format, origin_format)
    if data_format == "NDC1HWC0":
        shape_x = [shape_x[0] * shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5]]
        x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
        sum["format"] = "NDC1HWC0"
    else:
        if tbe_platform.api_check_support("tik.vcopy") and data_format == "NC1HWC0":
            shape_x = [shape_x[0], shape_x[1], 1, shape_x[2] * shape_x[3], shape_x[4]]
        x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())

    res = bn_training_reduce_compute(x_input, sum, square_sum,
                                     kernel_name=kernel_name)
    if data_format in ("NC1HWC0", "NDC1HWC0"):
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
    else:
        auto_sch_choose = set(shape_x) == {1}
        if auto_sch_choose:
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
        else:
            sch, tensor_list = bn_training_reduce_schedule_nd(res)
            with build_config():
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
            return
    tensor_list = [x_input] + list(res)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bn_training_reduce(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """
    try:
        bn_training_reduce_static(x, sum, square_sum, kernel_name)
    except:
        # when compile with static schedule failed, will use dynamic schedule
        x["range"] = gen_range(list(x.get("shape")))
        sum["range"] = gen_range(list(sum.get("shape")))
        square_sum["range"] = gen_range(list(square_sum.get("shape")))
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            bn_training_reduce_dynamic(x, sum, square_sum, kernel_name)
        else:
            with tbe_context.op_context.OpContext("static"):
                bn_training_reduce_dynamic(x, sum, square_sum, kernel_name)
