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
dynamic add_n
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
from impl.util import fusion_util
from impl.util import util_compute
from impl.util import util_common


# 'pylint: disable=invalid-name
def check_supported(inputs,
                    output,
                    N,
                    kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects,
        each with same shape, range and dtype of first input,
        only support float16, bfloat16, float32, int32, int64.
    output : dict
        shape, range and dtype of output,
        should be broadcast shape and type as input.
    N:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    len_tensor = len(inputs)
    is_unknown_shape = util_common.is_unknown([inputs[0]])
    shape = list(inputs[0].get("shape"))
    dtype = inputs[0].get("dtype")
    if not is_unknown_shape and len_tensor == 2 and shape == [6, 6] and dtype == "float32":
        return False, "not support"
    return True, ""


# 'pylint: disable=unused-argument,too-many-locals,redefined-argument-from-local,unused-variable,too-many-statements
# 'pylint: disable=invalid-name
def add_n_fusion_compute(datas, output, N, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders, all input data
    output : dict, dict of output
    N: nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : placeholder and res of the data's add_n
    """
    ph_datas = []
    for i, data_i in enumerate(datas):
        fusion_util.check_fusion_input([data_i])
        dict_data_i = fusion_util.extract_dict(data_i)
        shape_data_i = fusion_util.normalize_shape([dict_data_i])[0]
        ph_tmp = fusion_util.create_placeholder(data_i, shape_data_i)
        ph_datas.append(ph_tmp)
    res = ph_datas[0]
    for i, data_i in enumerate(ph_datas):
        if i == 0:
            continue
        res = tbe.vadd(res, data_i)
    return {"op_placeholder": ph_datas, "op_res": [res]}


# 'pylint: disable=invalid-name
@register_operator_compute("AddN", op_mode="dynamic", support_fusion=True)
def add_n_compute_for_batchmatmul(datas, output, N, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    N:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    # if fused with batch_matmul, split batch dim in batchmatmul
    batchmatmul_flag = False
    for i, data_n in enumerate(datas):
        batchmatmul_flag = util_compute.check_batchmatmul_fuse(data_n)
        if batchmatmul_flag:
            if "para_name" in data_n.op.attrs:
                para_name = data_n.op.attrs["para_name"]
                para_name += "_addn"
            else:
                para_name = "addn"
            batch_shape = shape_util.shape_to_list(data_n.op.attrs["batch_shape"])
            batchmatmul_node = data_n
            batchmatmul_idx = i
            shape_max = batch_shape + shape_util.shape_to_list(data_n.shape)[-4:]
            break
    nz_addn = []
    for i, data_n in enumerate(datas):
        if batchmatmul_flag and i != batchmatmul_idx:
            data_n = tbe.broadcast(data_n, shape_max)
            data_n = util_compute.batchmatmul_elem_reshape(batchmatmul_node, data_n, batch_shape, para_name + str(i))
            nz_addn.append(data_n)

    if not batchmatmul_flag:
        res = datas[0]
        format_out = ""
        for i, data_n in enumerate(datas):
            if i == 0:
                continue
            if "format" in data_n.op.attrs:
                format_out = data_n.op.attrs["format"]
            res = tbe.vadd(res, data_n)
        if format_out:
            res.op.attrs["format"] = format_out
        return res

    for i, nz_add in enumerate(nz_addn):
        batchmatmul_node = tbe.vadd(batchmatmul_node, nz_add)
    return batchmatmul_node


# 'pylint: disable=invalid-name
def add_n_compute(datas, output, N, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders, all input data
    output : dict, dict of output
    N: nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    data_type = datas[0].dtype
    has_covert_float32 = ((data_type == "float16" and tbe_platform.api_check_support("te.lang.cce.vadd", "float32")) or
                    (data_type == "bfloat16" and not tbe_platform.api_check_support("tbe.dsl.vadd", "bfloat16") and \
                tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")))

    first_data = datas[0] if not has_covert_float32 else tbe.cast_to(datas[0], "float32")
    res = first_data

    for i, data_i in enumerate(datas):
        if i == 0:
            continue
        tmp_data = data_i if not has_covert_float32 else tbe.cast_to(data_i, "float32")
        res = tbe.vadd(res, tmp_data)

    if has_covert_float32:
        if data_type == "float16":
            res = tbe.cast_to(res, "float16")
        else:
            res = tbe.round(res, "bfloat16")

    return res


# 'pylint: disable=invalid-name
@register_operator("AddN")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def add_n(inputs, output, N, kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects,
        each with same shape, range and dtype of first input,
        only support float16, bfloat16, float32, int32, int64.
    output : dict
        shape, range and dtype of output,
        should be broadcast shape and type as input.
    N:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    # check inputs num
    tensor_num = len(inputs) if N is None else N
    input_num = len(inputs)
    if input_num < 2:
        error_manager_vector.raise_err_input_param_range_invalid("add_n", "input_num", "8", "2", str(input_num))
    if input_num != tensor_num:
        error_manager_vector.raise_err_inputs_shape_not_equal("add_n", "input_num", "tensor_num", str(input_num),
                                                              str(tensor_num), str(input_num))

    dtype_0 = inputs[0].get("dtype").lower()
    for index in range(0, tensor_num):
        shape_input = inputs[index].get("shape")
        para_check.check_shape(shape_input, param_name="inputs")
        dtype_input = inputs[index].get("dtype").lower()
        check_list = ("bfloat16", "float16", "float32", "int32", "int64")
        para_check.check_dtype(dtype_input, check_list, param_name="inputs")
        if dtype_input != dtype_0:
            error_manager_vector.raise_err_inputs_shape_not_equal("add_n", "inputs[" + str(index) + "]", "inputs[0]",
                                                                  str(dtype_input), str(dtype_0), str(dtype_input))

    ins = classify(inputs, OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for inputs in ins:
        with tbe.compute():
            shape_normlize = shape_util.variable_shape(inputs)
            datas = []
            for index, shape_index in enumerate(shape_normlize):
                datas.append(tvm.placeholder(shape_index, name=f"data_{index}", dtype=dtype_0))
            # add_n_compute
            res = add_n_compute(datas, output, kernel_name)
            datas.append(res)
            tensors.append(datas)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors, "build_args": {"enable_alloc_mem_by_use_count": True}}
    tbe.build(schedules, config)
