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
in_top_k
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import is_unknown_rank_input


# 'pylint: disable=too-many-lines,too-many-locals,too-many-statements,too-many-arguments,unused-argument
# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # size of useful UB buffer
    UB_SIZE_BYTES = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    DTYPE_INT32 = "int32"
    TILING_PARAMS_NUM = 8
    TILING_PARAM_DTYPE = DTYPE_INT32
    BYTE_INT32 = 4
    BYTE_FLOAT32 = 4
    BLOCK_SIZE = 32
    V_SIZE_BYTES = 256
    MAX_SHAPE_SIZE = 2**31 - 1
    FP_MIN = -3.40282346638528860E38

    BIG_DIM0_TYPE = 1
    BIG_DIM1_TYPE = 2
    OTHER_TYPE = 3
    CHECK_INF = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ["Ascend910B", "Ascend910_93"]


class GlobalVarGM:
    """GlobalVarGM Class Defination"""
    def __init__(self, tik_instance):
        """"
        __init__
        """
        self.predictions_gm = None
        self.targets_gm = None
        self.tensor_output_gm = None
        self.tiling_gm = tik_instance.Tensor(Constant.DTYPE_INT32, (Constant.TILING_PARAMS_NUM,),
                                             name="tiling_gm", scope=tik.scope_gm)

    def set_predictions_gm(self, predictions_gm):
        """"
        set_predictions_gm
        """
        self.predictions_gm = predictions_gm

    def get_predictions_gm(self):
        """"
        get_predictions_gm
        """
        return self.predictions_gm

    def set_targets_gm(self, targets_gm):
        """"
        set_targets_gm
        """
        self.targets_gm = targets_gm

    def get_targets_gm(self):
        """"
        get_targets_gm
        """
        return self.targets_gm

    def set_tensor_output_gm(self, tensor_output_gm):
        """"
        set_tensor_output_gm
        """
        self.tensor_output_gm = tensor_output_gm

    def get_tensor_output_gm(self):
        """"
        get_tensor_output_gm
        """
        return self.tensor_output_gm


class GlobalVarTilingScalar:
    """GlobalVarTilingScalar Class Defination"""
    def __init__(self, tik_instance, tiling_gm):
        """"
        __init__
        """
        self.num_rows_scalar = tik_instance.Scalar(dtype="int32", name="num_rows_scalar")
        self.num_cols_scalar = tik_instance.Scalar(dtype="int32", name="num_cols_scalar")
        self.num_cores_scalar = tik_instance.Scalar(dtype="int32", name="num_cores_scalar")
        self.tiling_ub = tik_instance.Tensor(dtype=Constant.TILING_PARAM_DTYPE,
                                             shape=(Constant.TILING_PARAMS_NUM,),
                                             name="tiling_ub",
                                             scope=tik.scope_ubuf)
        #mov tiling params from gm to ub
        tik_instance.data_move(self.tiling_ub, tiling_gm, 0, 1,
                               Constant.TILING_PARAMS_NUM * Constant.BYTE_INT32 // Constant.BLOCK_SIZE, 0, 0)
        # input scalar in flowtable
        input_scalar_index = 0
        self.num_rows_scalar.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.num_cols_scalar.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.num_cores_scalar.set_as(self.tiling_ub[input_scalar_index])
        self.k_value_index = input_scalar_index + 1

    def get_rows_num(self):
        """"
        get_rows_num
        """
        return self.num_rows_scalar

    def get_cols_num(self):
        """"
        get_cols_num
        """
        return self.num_cols_scalar

    def get_core_num(self):
        """"
        get_core_num
        """
        return self.num_cores_scalar

    def get_k_value(self, tik_instance):
        k_scalar = tik_instance.Scalar(dtype="int32", name="k_scalar")
        k_scalar.set_as(self.tiling_ub[self.k_value_index])
        return k_scalar


@register_operator("InTopKD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def in_top_k(predictions, targets, precision, k, kernel_name="in_top_k"):
    """"
    the main function of in_top_k

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of precision
             include keys(shape and dtype)
    precision: dict of precision
               reserved output
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    ----------
    tik_instance: tik_instance
    """
    Constant.UB_SIZE_BYTES = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    Constant.CHECK_INF = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ["Ascend910B", "Ascend910_93"]
    prediction_dtype = predictions.get("dtype").lower()
    target_dtype = "int32"
    tik_instance = tik.Tik()
    obj_gm = GlobalVarGM(tik_instance)
    obj_tiling_gm = obj_gm.tiling_gm
    obj_tiling = GlobalVarTilingScalar(tik_instance, obj_tiling_gm)

    row = obj_tiling.get_rows_num()
    column = obj_tiling.get_cols_num()

    para_check.check_dtype(prediction_dtype, ('float32',), param_name="predictions")
    para_check.check_dtype(target_dtype, ('int32',), param_name="targets")

    if not tbe_platform.api_check_support("tik.vconv", "f322f16"):
        error_manager_vector.raise_err_specific_reson(kernel_name, "this product does not supported float32")

    _check_input_shape(predictions, targets, kernel_name)

    # the number of elements on a block.
    block_element = Constant.BLOCK_SIZE // Constant.BYTE_FLOAT32
    column_aligned = ((column + block_element - 1) // block_element * block_element)
    #the max size of a tensor on ub in the compute schedule of in_top_k.
    coexisting_tensor_num = 5
    max_tensor_size = Constant.UB_SIZE_BYTES // coexisting_tensor_num
    # the number of rows UB can deal each time.
    core_row_capicity = max_tensor_size // (column_aligned * Constant.BYTE_FLOAT32)

    prediction_tensor = tik_instance.Tensor(dtype=prediction_dtype,
                                            shape=(Constant.MAX_SHAPE_SIZE,),
                                            name="prediction_tensor",
                                            scope=tik.scope_gm)
    target_tensor = tik_instance.Tensor(dtype=target_dtype,
                                        shape=(Constant.MAX_SHAPE_SIZE,),
                                        name="target_tensor",
                                        scope=tik.scope_gm)
    output_tensor = tik_instance.Tensor(dtype="uint8",
                                        shape=(Constant.MAX_SHAPE_SIZE,),
                                        name="output_tensor",
                                        scope=tik.scope_gm)
    obj_gm.set_predictions_gm(prediction_tensor)
    obj_gm.set_targets_gm(target_tensor)
    obj_gm.set_tensor_output_gm(output_tensor)
    mini_cloud_core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    tbe_context.get_context().add_compile_info("vars", {
        "mini_cloud_core_nums": mini_cloud_core_nums
    })

    if k is None:
        k = obj_tiling.get_k_value(tik_instance)
    real_core_num = obj_tiling.get_core_num()

    with tik_instance.if_scope(k <= 0):
        _in_top_k_special_k(tik_instance, obj_tiling, obj_gm, k, targets)
    with tik_instance.else_scope():
        with tik_instance.if_scope(k >= column):
            _in_top_k_special_k(tik_instance, obj_tiling, obj_gm, k, targets)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, mini_cloud_core_nums, block_num=mini_cloud_core_nums) as core_loop:
                with tik_instance.if_scope(core_loop < real_core_num):
                    shape_info = {
                        "core_loop": core_loop,
                        "mini_cloud_core_nums": mini_cloud_core_nums
                    }
                    with tik_instance.if_scope(core_row_capicity > 0):
                        with tik_instance.if_scope(row <= Constant.BLOCK_SIZE):
                            _in_top_k_single_core(tik_instance, shape_info, obj_tiling, obj_gm, k, targets)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(core_row_capicity < Constant.BLOCK_SIZE):
                                _in_top_k_mul_core_v2(tik_instance, shape_info, obj_tiling, obj_gm, k, targets)
                            with tik_instance.else_scope():
                                _in_top_k_mul_core(tik_instance, shape_info, obj_tiling, obj_gm, k, targets)
                    with tik_instance.else_scope():
                        _in_top_k_tiling_column(tik_instance, shape_info, obj_tiling, obj_gm, k, targets)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[obj_gm.get_predictions_gm(), obj_gm.get_targets_gm()],
                          outputs=[obj_gm.get_tensor_output_gm()], flowtable=(obj_gm.tiling_gm,))
    return tik_instance


def _check_input_shape(predictions, targets, kernel_name):
    """
    check predictions_shape and targets_shape
    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of precision
             include keys(shape and dtype)
    kernel_name: kernel_name
    Returns
    -------
    None
    """
    if is_unknown_rank_input([predictions, targets]):
        return
    predictions_shape = predictions.get("shape")
    targets_shape = targets.get("shape")
    # the predictions is 2-dimensional
    if len(predictions_shape) != 2:
        error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'predictions', 2, 2,
                                                                 len(predictions_shape))
    # the targets is 1-dimensional
    if len(targets_shape) != 1:
        error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'targets', 1, 1, len(targets_shape))

    if (predictions_shape[0] != -1 and targets_shape[0] != -1) and (predictions_shape[0] != targets_shape[0]):
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "First dimension of predictions must match the length of targets."
        )


def _in_top_k_special_k(tik_instance, obj_tiling, obj_gm, k, targets):
    """"
    the _in_top_k_special_k function of the in_top_k

    Parameters
    ----------
    tik_instance: tik_instance
    obj_tiling: obj_tiling
                include keys(num_rows_scalar, num_cols_scalar, num_cores_scalar)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    row = obj_tiling.get_rows_num()
    column = obj_tiling.get_cols_num()

    target_dtype = targets.get('dtype').lower()
    precision_dtype = obj_gm.get_tensor_output_gm().dtype

    element = 1
    if isinstance(k, int) and k <= 0:
        element = 0
    else:
        element = tik_instance.Scalar(dtype="float16", name="element_scalar")
        with tik_instance.if_scope(k <= 0):
            element.set_as(0)
        with tik_instance.else_scope():
            element.set_as(1)

    target_ub = tik_instance.Tensor(dtype=target_dtype,
                                    shape=(Constant.BLOCK_SIZE,),
                                    name="target_ub",
                                    scope=tik.scope_ubuf)
    output_ub = tik_instance.Tensor(dtype=precision_dtype,
                                    shape=(Constant.BLOCK_SIZE,),
                                    name="output_ub",
                                    scope=tik.scope_ubuf)
    tensor_ub = tik_instance.Tensor(dtype="float16",
                                    shape=(Constant.BLOCK_SIZE,),
                                    name="tensor_ub",
                                    scope=tik.scope_ubuf)

    tik_instance.vector_dup(Constant.BLOCK_SIZE, tensor_ub, element, 1, 1, 1)
    tensor_ub_sel = tik_instance.Tensor(dtype="float16",
                                        shape=(Constant.BLOCK_SIZE,),
                                        name="tensor_ub_sel",
                                        scope=tik.scope_ubuf)
    dst_ub = tik_instance.Tensor(dtype="int32",
                                 shape=(Constant.BLOCK_SIZE,),
                                 name="dst_ub",
                                 scope=tik.scope_ubuf)
    dst_ub1 = tik_instance.Tensor(dtype="float16",
                                  shape=(Constant.BLOCK_SIZE,),
                                  name="dst_ub1",
                                  scope=tik.scope_ubuf)
    tensor_zeros = tik_instance.Tensor(dtype="float16",
                                       shape=(Constant.BLOCK_SIZE,),
                                       name="tensor_zeros",
                                       scope=tik.scope_ubuf)
    tik_instance.vector_dup(Constant.BLOCK_SIZE, tensor_zeros, 0, 1, 1, 1)

    with tik_instance.if_scope(k <= 0):
        tik_instance.data_move(target_ub, obj_gm.get_targets_gm(), 0, 1, 1, 0, 0)
        tik_instance.vconv(Constant.BLOCK_SIZE, '', output_ub, tensor_ub, 1, 1, 1, 8, 8)
    copy_repeat_times = (row + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE
    with tik_instance.for_range(0, copy_repeat_times) as i:
        with tik_instance.if_scope(k >= column):
            tik_instance.data_move(target_ub, obj_gm.get_targets_gm()[i * Constant.BLOCK_SIZE], 0, 1, 4, 0, 0)
            invalid_mask = calc_invalid_mask(tik_instance, target_ub, Constant.BLOCK_SIZE, column, tensor_zeros,
                                             dst_ub, dst_ub1, 0)
            tik_instance.vsel(Constant.BLOCK_SIZE, 0, tensor_ub_sel, invalid_mask, tensor_zeros, tensor_ub, 1, 1, 1, 1)
            invalid_mask = calc_invalid_mask(tik_instance, target_ub, Constant.BLOCK_SIZE, column, tensor_zeros,
                                             dst_ub, dst_ub1, 1)
            tik_instance.vsel(Constant.BLOCK_SIZE, 0, tensor_ub_sel, invalid_mask,
                              tensor_zeros, tensor_ub_sel, 1, 1, 1, 1)
            tik_instance.vconv(Constant.BLOCK_SIZE, '', output_ub, tensor_ub_sel, 1, 1, 1, 8, 8)
        tik_instance.data_move(obj_gm.get_tensor_output_gm()[i * Constant.BLOCK_SIZE], output_ub, 0, 1, 1, 0, 0)


def calc_invalid_mask(tik_instance, target_ub, mask_len, column, tensor_zero, dst_ub, dst_ub1, flag):
    """"
    the calc_invalid_mask function

    Parameters
    ----------
    tik_instance: tik_instance
    target_ub: dict of target_ub
               include keys(shape and dtype)
    mask_len: mask_len
    column: column
    src: dict of src
         include keys(shape and dtype)
    dst_ub: dict of dst_ub
            include keys(shape and dtype)
    dst_ub1: dict of dst_ub1
             include keys(shape and dtype)
    flag: 0:  target_ub >= column
          1:  target_ub < 0
    Returns
    ----------
    mask: cmpmask
    """
    repeat_time = 1
    deq_scale = 1.0
    if flag == 0:
        src1 = tik_instance.Tensor(dtype="int32",
                                   shape=(128,),
                                   name="src1",
                                   scope=tik.scope_ubuf)
        tik_instance.vector_dup(mask_len, src1, column, 1, 1, 8)
        tik_instance.vsub(mask_len, dst_ub, target_ub, src1, 1, 1, 1, 1, 8, 8, 8)
        tik_instance.vconv(mask_len, 'none', dst_ub1, dst_ub, repeat_time, 1, 1, 4, 8, deq_scale)
        mask = tik_instance.vcmp_ge(mask_len, dst_ub1, tensor_zero, 1, 0)
    else:
        target_conv_ub = tik_instance.Tensor(dtype="float16",
                                             shape=(128,),
                                             name="target_conv_ub",
                                             scope=tik.scope_ubuf)
        tik_instance.vconv(mask_len, 'none', target_conv_ub, target_ub, repeat_time, 1, 1, 4, 8, deq_scale)
        mask = tik_instance.vcmp_lt(mask_len, target_conv_ub, tensor_zero, 1, 0)
    return mask


def _in_top_k_single_core(tik_instance, shape_info, obj_tiling, obj_gm, k, targets):
    """"
    the _in_top_k_single_core function

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include key(core_loop, mini_cloud_core_nums)
    obj_tiling: obj_tiling
                include keys(num_rows_scalar, num_cols_scalar, num_cores_scalar)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    element_bytes = 4
    block_element = Constant.BLOCK_SIZE // element_bytes
    row = obj_tiling.get_rows_num()
    column = obj_tiling.get_cols_num()
    coexisting_tensor_num = 5
    column_aligned = ((column + block_element - 1) // block_element * block_element)
    max_tensor_size = Constant.UB_SIZE_BYTES // coexisting_tensor_num
    core_row_capicity = max_tensor_size // (column_aligned * element_bytes)
    split_row_times = (row + core_row_capicity - 1) // core_row_capicity
    row_nums = (row + split_row_times - 1) // split_row_times
    row_remainder = row - (split_row_times - 1) * row_nums
    core_nums = split_row_times
    split_rows_nums = 0
    shape_info["targets_dtype"] = targets.get("dtype").lower()

    # copy predictions to ub from gm.
    # if the last dimension is not divided by 32 bytes, just aligned.
    with tik_instance.for_range(0, split_row_times) as outer_loop:
        shape_info["outer_loop"] = outer_loop
        shape_info["index"] = outer_loop * row_nums
        shape_info["core_nums"] = core_nums
        shape_info["row_remainder"] = row_remainder
        shape_info["row_nums"] = row_nums
        shape_info["column"] = column
        shape_info["column_aligned"] = column_aligned
        shape_info["split_rows_nums"] = split_rows_nums
        shape_info["max_tensor_size"] = max_tensor_size

        length = tik_instance.Scalar("int64")
        length.set_as(row_nums + Constant.BLOCK_SIZE - 1)
        with tik_instance.if_scope(column_aligned <= 64):
            tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, Constant.BIG_DIM0_TYPE)
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[outer_loop * row_nums], tensor_output_ub,
                                   0, 1, length // Constant.BLOCK_SIZE, 0, 0)
        with tik_instance.else_scope():
            with tik_instance.if_scope(column_aligned <= 4096):
                tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, Constant.BIG_DIM1_TYPE)
                tik_instance.data_move(obj_gm.get_tensor_output_gm()[outer_loop * row_nums], tensor_output_ub,
                                       0, 1, length // Constant.BLOCK_SIZE, 0, 0)
            with tik_instance.else_scope():
                tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, Constant.OTHER_TYPE)
                tik_instance.data_move(obj_gm.get_tensor_output_gm()[outer_loop * row_nums], tensor_output_ub,
                                       0, 1, length // Constant.BLOCK_SIZE, 0, 0)


def _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, mask):
    """"
    the _in_top_k_inter_process function

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include keys(core_loop, mini_cloud_core_nums, outer_loop, index,
                             core_nums, row_remainder, row_nums, column,
                             column_aligned, split_rows_nums, max_tensor_size)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    outer_loop = shape_info.get("outer_loop")
    index1 = shape_info.get("index")
    core_nums = shape_info.get("core_nums")
    row_remainder = shape_info.get("row_remainder")
    row_nums = shape_info.get("row_nums")
    column = shape_info.get("column")
    column_aligned = shape_info.get("column_aligned")
    prediction_tensor = obj_gm.get_predictions_gm()
    target_tensor = obj_gm.get_targets_gm()
    max_tensor_size = shape_info.get("max_tensor_size")

    half_mask_value = 64
    mask_value = 128
    element_bytes = 4
    carry = 2
    block_element = Constant.BLOCK_SIZE // element_bytes
    prediction_dtype = "float32"
    target_dtype = shape_info.get("targets_dtype")
    core_row_num = tik_instance.Scalar("int64")

    k_conv_dtype = tik_instance.Scalar("float32")
    k_conv_dtype.set_as(k)

    # step 0: set some shape value of tensor in UB.
    with tik_instance.if_scope(outer_loop < core_nums - 1):
        core_row_num.set_as(row_nums)
    with tik_instance.else_scope():
        core_row_num.set_as(row_remainder)

    prediction_tensor_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                               shape=(max_tensor_size // element_bytes,),
                                               name="prediction_tensor_ub",
                                               scope=tik.scope_ubuf)

    if mask == Constant.BIG_DIM0_TYPE:
        target_shape = (max_tensor_size // element_bytes // 8,)
    elif mask == Constant.BIG_DIM1_TYPE:
        target_shape = (max_tensor_size // element_bytes // 72,)
    else:
        target_shape = (Constant.BLOCK_SIZE,)

    target_ub = tik_instance.Tensor(dtype=target_dtype,
                                    shape=target_shape,
                                    name="target_ub",
                                    scope=tik.scope_ubuf)
    tik_instance.data_move(target_ub, target_tensor[index1], 0, 1, (core_row_num + block_element - 1) // block_element,
                           0, 0)

    with tik_instance.if_scope(column % block_element == 0):
        length = tik_instance.Scalar("int64")
        length.set_as(core_row_num * column_aligned)
        index = index1 * column_aligned
        tik_instance.data_move(prediction_tensor_ub, prediction_tensor[index], 0, 1, length // block_element, 0, 0)
    with tik_instance.else_scope():
        index = index1 * column
        with tik_instance.new_stmt_scope(disable_sync=True):
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.data_move(prediction_tensor_ub[i * column_aligned], prediction_tensor[index + i * column],
                                       0, 1, column_aligned // block_element, 0, 0)

        # dirty data set as FP_MIN, for example, A[:,1:15] is target data, then set A[:,15:16] as FP_MIN.
        reg_data = tik_instance.Scalar(prediction_dtype)
        reg_data.set_as(Constant.FP_MIN)
        with tik_instance.new_stmt_scope(disable_sync=True):
            with tik_instance.for_range(0, core_row_num) as i:
                with tik_instance.for_range(0, column_aligned - column) as j:
                    prediction_tensor_ub[i * column_aligned + column + j] = reg_data

    # step 1: index the predictions's elements according to the targets.
    # the result is in tensor data_ub.
    tensor_size = row_nums * column_aligned
    data_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                  shape=(max_tensor_size // element_bytes,),
                                  name="data_ub",
                                  scope=tik.scope_ubuf)
    # set the number of repeat.
    column_reduce_times = column_aligned // half_mask_value
    column_reduce_remainder = column_aligned % half_mask_value

    if Constant.CHECK_INF:
        zero_mul_reps = max_tensor_size // element_bytes // half_mask_value
        zero_mul_rems = max_tensor_size // element_bytes % half_mask_value
        with tik_instance.if_scope(zero_mul_reps > 0):
            tik_instance.vmuls(half_mask_value, data_ub, prediction_tensor_ub, 0.0, zero_mul_reps, 1, 1, 8, 8)
            tik_instance.vadd(half_mask_value,
                              prediction_tensor_ub, data_ub, prediction_tensor_ub,
                              zero_mul_reps, 1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(zero_mul_rems != 0):
            tik_instance.vmuls(zero_mul_rems,
                               data_ub[zero_mul_reps * half_mask_value],
                               prediction_tensor_ub[zero_mul_reps * half_mask_value],
                               0.0, 1, 1, 1, 8, 8)
            tik_instance.vadd(zero_mul_rems,
                              prediction_tensor_ub[zero_mul_reps * half_mask_value],
                              data_ub[zero_mul_reps * half_mask_value],
                              prediction_tensor_ub[zero_mul_reps * half_mask_value],
                              1, 1, 1, 1, 8, 8, 8)

    with tik_instance.if_scope(column_reduce_times > 0):
        with tik_instance.new_stmt_scope(disable_sync=True):
            with tik_instance.for_range(0, core_row_num) as i:
                scalar_target = tik_instance.Scalar(target_dtype, "scalar_target")
                scalar_target.set_as(target_ub[i])
                with tik_instance.if_scope(tik.any(scalar_target < 0, scalar_target >= column)):
                    scalar_target.set_as(0)
                scalar_value = tik_instance.Scalar(prediction_dtype)
                scalar_value.set_as(prediction_tensor_ub[i * column_aligned + scalar_target])
                tik_instance.vector_dup(half_mask_value, data_ub[i * column_aligned], scalar_value,
                                        column_reduce_times, 1, 8, 0)
                with tik_instance.if_scope(column_reduce_remainder != 0):
                    tik_instance.vector_dup(column_reduce_remainder,
                                            data_ub[i * column_aligned + column_reduce_times * half_mask_value],
                                            scalar_value, 1, 1, 1, 0)

    with tik_instance.else_scope():
        with tik_instance.new_stmt_scope(disable_sync=True):
            with tik_instance.for_range(0, core_row_num) as i:
                scalar_target = tik_instance.Scalar(target_dtype, "scalar_target")
                scalar_target.set_as(target_ub[i])
                with tik_instance.if_scope(tik.any(scalar_target < 0, scalar_target >= column)):
                    scalar_target.set_as(0)
                scalar_value = tik_instance.Scalar(prediction_dtype)
                scalar_value.set_as(prediction_tensor_ub[i * column_aligned + scalar_target])
                tik_instance.vector_dup(column_reduce_remainder, data_ub[i * column_aligned], scalar_value, 1, 1, 1, 0)

    if tbe_platform.api_check_support("tik.vcmp_gt", "float32"):
        data_zeros = tik_instance.Tensor(dtype="float32",
                                         shape=(half_mask_value, 1),
                                         name="data_zeros",
                                         scope=tik.scope_ubuf)
        tik_instance.vector_dup(half_mask_value, data_zeros, 0, 1, 1, 1, 0)
        data_ones = tik_instance.Tensor(dtype="float32",
                                        shape=(half_mask_value, 1),
                                        name="data_ones",
                                        scope=tik.scope_ubuf)
        tik_instance.vector_dup(half_mask_value, data_ones, 1, 1, 1, 1, 0)

        data_sign = tik_instance.Tensor(dtype="float32",
                                        shape=(max_tensor_size // element_bytes,),
                                        name="data_sign",
                                        scope=tik.scope_ubuf)
        repeat_times = tensor_size // half_mask_value
        tail_mask = tensor_size % half_mask_value
        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vector_dup(half_mask_value, data_sign, 0, repeat_times, 1, 8)
        with tik_instance.if_scope(tail_mask != 0):
            tik_instance.vector_dup(tail_mask, data_sign[repeat_times * half_mask_value], 0, 1, 1, 8)

        if Constant.CHECK_INF:
            data_cmpmask = tik_instance.Tensor(dtype="uint64",
                                               shape=(4, ),
                                               name="data_cmpmask",
                                               scope=tik.scope_ubuf)
            tik_instance.vector_dup(8, data_cmpmask.reinterpret_cast_to("uint32"), 0, 1, 1, 1, 0)
            scalar_nan_check_mask = tik_instance.Scalar("uint64", "scalar_nan_check_mask")
            scalar_tail_mask = tik_instance.Scalar("uint64", "scalar_tail_mask", init_value=1)
            # bit shift: scalar_tail_mask = 1 << column_reduce_remainder
            with tik_instance.for_range(0, column_reduce_remainder) as i:
                scalar_tail_mask = scalar_tail_mask * 2
            scalar_tail_mask = ~ (scalar_tail_mask - 1)
            tmp_reduce_data_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                               shape=(half_mask_value, 1),
                                               name="tmp_reduce_data_ub",
                                               scope=tik.scope_ubuf)

            with tik_instance.if_scope(column_reduce_times > 0):
                with tik_instance.new_stmt_scope(disable_sync=False):
                    with tik_instance.for_range(0, core_row_num) as i:
                        tik_instance.vector_dup(half_mask_value, tmp_reduce_data_ub, 0.0, 1, 1, 8)
                        with tik_instance.for_range(0, column_reduce_times) as j:
                            tik_instance.vmax(half_mask_value,
                                tmp_reduce_data_ub,
                                prediction_tensor_ub[i * column_aligned + half_mask_value * j],
                                tmp_reduce_data_ub,
                                1, 1, 1, 1, 8, 8, 8)
                        with tik_instance.if_scope(column_reduce_remainder != 0):
                            tik_instance.vmax(column_reduce_remainder,
                                tmp_reduce_data_ub,
                                prediction_tensor_ub[i * column_aligned + column_reduce_times * half_mask_value],
                                tmp_reduce_data_ub,
                                1, 1, 1, 1, 8, 8, 8)
                        nan_check = tik_instance.vcmp_eq(half_mask_value,
                            tmp_reduce_data_ub,
                            tmp_reduce_data_ub,
                            1, 1)
                        tik_instance.mov_cmpmask_to_tensor(data_cmpmask, nan_check)
                        scalar_nan_check_mask.set_as(data_cmpmask[0])

                        with tik_instance.if_scope(scalar_nan_check_mask == 0xffffffffffffffff):
                            with tik_instance.for_range(0, column_reduce_times) as j:
                                srcmask = tik_instance.vcmp_gt(half_mask_value,
                                                prediction_tensor_ub[i * column_aligned + half_mask_value * j],
                                                data_ub[i * column_aligned + half_mask_value * j],
                                                1, 1)
                                tik_instance.vsel(half_mask_value, 0,
                                                  data_sign[i * column_aligned + half_mask_value * j],
                                                  srcmask, data_ones, data_zeros,
                                                  1, 1, 1, 1)
                            with tik_instance.if_scope(column_reduce_remainder != 0):
                                srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                        prediction_tensor_ub[i * column_aligned +
                                                                column_reduce_times * half_mask_value],
                                                        data_ub[i * column_aligned +
                                                                column_reduce_times * half_mask_value],
                                                        1, 1)
                                tik_instance.vsel(column_reduce_remainder, 0,
                                                data_sign[i * column_aligned + column_reduce_times * half_mask_value],
                                                srcmask, data_ones, data_zeros, 1, 1, 1, 1)
                        with tik_instance.else_scope():
                            data_sign[i * column_aligned] = k_conv_dtype

            with tik_instance.else_scope():
                with tik_instance.new_stmt_scope(disable_sync=False):
                    with tik_instance.for_range(0, core_row_num) as i:
                        nan_check = tik_instance.vcmp_eq(column_reduce_remainder,
                            prediction_tensor_ub[i * column_aligned],
                            prediction_tensor_ub[i * column_aligned], 1, 1)
                        tik_instance.mov_cmpmask_to_tensor(data_cmpmask, nan_check)
                        scalar_nan_check_mask.set_as(data_cmpmask[0])
                        scalar_nan_check_mask = scalar_nan_check_mask | scalar_tail_mask

                        with tik_instance.if_scope(scalar_nan_check_mask == 0xffffffffffffffff):
                            srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                           prediction_tensor_ub[i * column_aligned],
                                                           data_ub[i * column_aligned],
                                                           1, 1)
                            tik_instance.vsel(column_reduce_remainder, 0,
                                              data_sign[i * column_aligned],
                                              srcmask, data_ones, data_zeros,
                                              1, 1, 1, 1)
                        with tik_instance.else_scope():
                            data_sign[i * column_aligned] = k_conv_dtype
        else:
            with tik_instance.if_scope(column_reduce_times > 0):
                with tik_instance.new_stmt_scope(disable_sync=True):
                    with tik_instance.for_range(0, core_row_num) as i:
                        with tik_instance.for_range(0, column_reduce_times) as j:
                            srcmask = tik_instance.vcmp_gt(half_mask_value,
                                                        prediction_tensor_ub[i * column_aligned + half_mask_value * j],
                                                        data_ub[i * column_aligned + half_mask_value * j], 1, 1)
                            tik_instance.vsel(half_mask_value, 0, data_sign[i * column_aligned + half_mask_value * j],
                                            srcmask, data_ones, data_zeros, 1, 1, 1, 1)
                        with tik_instance.if_scope(column_reduce_remainder != 0):
                            srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                        prediction_tensor_ub[i * column_aligned +
                                                                                column_reduce_times * half_mask_value],
                                                        data_ub[i * column_aligned +
                                                                column_reduce_times * half_mask_value],
                                                        1, 1)
                            tik_instance.vsel(column_reduce_remainder, 0,
                                            data_sign[i * column_aligned + column_reduce_times * half_mask_value],
                                            srcmask, data_ones, data_zeros, 1, 1, 1, 1)
            with tik_instance.else_scope():
                with tik_instance.new_stmt_scope(disable_sync=True):
                    with tik_instance.for_range(0, core_row_num) as i:
                        srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                       prediction_tensor_ub[i * column_aligned],
                                                       data_ub[i * column_aligned], 1, 1)
                        tik_instance.vsel(column_reduce_remainder, 0, data_sign[i * column_aligned],
                                        srcmask, data_ones,
                                        data_zeros, 1, 1, 1, 1)

        mid_result_ub = data_sign
    else:
        # step 2, prediction_tensor subtract data_ub.
        column_remainder = column % half_mask_value
        reduce_times = column // half_mask_value
        repeat_times = (carry * tensor_size) // mask_value
        tail_mask = (carry * tensor_size) % mask_value
        half_sub = tik_instance.Tensor(dtype="float16",
                                       shape=(max_tensor_size // element_bytes * carry,),
                                       name="half_sub",
                                       scope=tik.scope_ubuf)

        with tik_instance.if_scope(reduce_times > 0):
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    tik_instance.vsub(half_mask_value, prediction_tensor_ub[i * column_aligned],
                                      prediction_tensor_ub[i * column_aligned], data_ub[i * column_aligned],
                                      reduce_times, 1, 1, 1, 8, 8, 8)
                    with tik_instance.if_scope(column_remainder != 0):
                        index = reduce_times * half_mask_value + i * column_aligned
                        tik_instance.vsub(column_remainder, prediction_tensor_ub[index], prediction_tensor_ub[index],
                                          data_ub[index], 1, 1, 1, 1, 8, 8, 8)
        with tik_instance.else_scope():
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    tik_instance.vsub(column_remainder, prediction_tensor_ub[i * column_aligned],
                                      prediction_tensor_ub[i * column_aligned], data_ub[i * column_aligned],
                                      1, 1, 1, 1, 8, 8, 8)

        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vector_dup(mask_value, half_sub, 0, repeat_times, 1, 8)
        with tik_instance.if_scope(tail_mask != 0):
            tik_instance.vector_dup(tail_mask, half_sub[repeat_times * mask_value], 0, 1, 1, 8)

        with tik_instance.if_scope(column_reduce_times > 0):
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    tik_instance.vconv(half_mask_value, '', half_sub[i * column_aligned * carry],
                                       prediction_tensor_ub[i * column_aligned], column_reduce_times, 1, 1, 4, 8)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                with tik_instance.new_stmt_scope(disable_sync=True):
                    with tik_instance.for_range(0, core_row_num) as i:
                        tik_instance.vconv(column_reduce_remainder, '',
                                           half_sub[column_reduce_times * half_mask_value +
                                                    carry * i * column_aligned],
                                           prediction_tensor_ub[i * column_aligned +
                                                                column_reduce_times * half_mask_value],
                                           1, 1, 1, 4, 8)
        with tik_instance.else_scope():
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    tik_instance.vconv(column_reduce_remainder, '', half_sub[carry * i * column_aligned],
                                       prediction_tensor_ub[i * column_aligned], 1, 1, 1, 4, 8)

        # step 3, if half_sub[i, j] > 0, then the according element data_sign[i, j] set 1, else 0.
        column_reduce_times = column_aligned // mask_value
        column_reduce_remainder = column_aligned % mask_value

        data_sign = tik_instance.Tensor(dtype="float16",
                                        shape=(max_tensor_size // element_bytes * carry,),
                                        name="data_sign",
                                        scope=tik.scope_ubuf)
        data_zeros = tik_instance.Tensor(dtype="float16",
                                         shape=(mask_value, 1),
                                         name="data_zeros",
                                         scope=tik.scope_ubuf)
        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vector_dup(mask_value, data_sign, 0, repeat_times, 1, 8)
        with tik_instance.if_scope(tail_mask != 0):
            tik_instance.vector_dup(tail_mask, data_sign[repeat_times * mask_value], 0, 1, 1, 8)

        tik_instance.vector_dup(mask_value, data_zeros, 0, 1, 1, 1, 0)
        data_ones = tik_instance.Tensor(dtype="float16",
                                        shape=(mask_value, 1),
                                        name="data_ones",
                                        scope=tik.scope_ubuf)
        tik_instance.vector_dup(mask_value, data_ones, 1, 1, 1, 1, 0)

        with tik_instance.if_scope(column_reduce_times > 0):
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    with tik_instance.for_range(0, column_reduce_times) as j:
                        srcmask = tik_instance.vcmp_gt(mask_value,
                                                       half_sub[i * column_aligned * carry + mask_value * j],
                                                       data_zeros, 1, 1)
                        tik_instance.vsel(mask_value, 0, data_sign[i * column_aligned * carry + mask_value * j],
                                          srcmask, data_ones, data_zeros, 1, 1, 1, 1)
                    with tik_instance.if_scope(column_reduce_remainder != 0):
                        srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                       half_sub[i * column_aligned * carry +
                                                                column_reduce_times * mask_value],
                                                       data_zeros, 1, 1)
                        tik_instance.vsel(column_reduce_remainder, 0,
                                          data_sign[i * column_aligned * carry + column_reduce_times * mask_value],
                                          srcmask, data_ones, data_zeros, 1, 1, 1, 1)
        with tik_instance.else_scope():
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    srcmask = tik_instance.vcmp_gt(column_reduce_remainder, half_sub[i * column_aligned * carry],
                                                   data_zeros, 1, 1)
                    tik_instance.vsel(column_reduce_remainder, 0, data_sign[i * column_aligned * carry],
                                      srcmask, data_ones, data_zeros, 1, 1, 1, 1)

        # step 4: do reduce sum in each row of data_sign to count the number which larger than the element indexing from
        # the target.
        column_reduce_times = column_aligned // half_mask_value
        column_reduce_remainder = column_aligned % half_mask_value

        with tik_instance.if_scope(column_reduce_times > 0):
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, core_row_num) as i:
                    tik_instance.vconv(half_mask_value, '', prediction_tensor_ub[i * column_aligned],
                                       data_sign[i * column_aligned * carry], column_reduce_times, 1, 1, 8, 4)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                with tik_instance.new_stmt_scope(disable_sync=True):
                    with tik_instance.for_range(0, core_row_num) as i:
                        tik_instance.vconv(column_reduce_remainder, '',
                                           prediction_tensor_ub[column_reduce_times * half_mask_value +
                                                                i * column_aligned],
                                           data_sign[carry * i * column_aligned +
                                                     column_reduce_times * half_mask_value],
                                           1, 1, 1, 8, 4)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vconv(column_reduce_remainder, '', prediction_tensor_ub[i * column_aligned],
                                   data_sign[carry * i * column_aligned], 1, 1, 1, 8, 4)

        mid_result_ub = prediction_tensor_ub

    reduce_mask = tik_instance.Scalar(dtype="int64", name="reduce_mask")
    with tik_instance.if_scope(column_reduce_remainder != 0):
        reduce_mask.set_as(column_reduce_times + 1)
    with tik_instance.else_scope():
        reduce_mask.set_as(column_reduce_times)

    data_bool = tik_instance.Tensor(dtype="float32",
                                    shape=target_shape,
                                    name="data_bool",
                                    scope=tik.scope_ubuf)

    if mask == Constant.BIG_DIM0_TYPE:
        with tik_instance.if_scope(column_reduce_remainder != 0):
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(column_reduce_remainder, data_bool[i], mid_result_ub[i * column_aligned], 1,
                                   1, 1, 1, 0)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value, data_bool[i], mid_result_ub[i * column_aligned], 1, 1, 1, 1,
                                   0)
    elif mask == Constant.BIG_DIM1_TYPE:
        tensor_sec_reduce = tik_instance.Tensor(dtype="float32",
                                                shape=(max_tensor_size // element_bytes // 72, half_mask_value),
                                                name="tensor_sec_reduce",
                                                scope=tik.scope_ubuf)
        zeros_init = tik_instance.Scalar("float32")
        zeros_init.set_as(0)
        with tik_instance.if_scope(column_reduce_remainder != 0):
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce, zeros_init, row_nums, 1, 8, 0)
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce[i * half_mask_value],
                                   mid_result_ub[i * column_aligned], column_reduce_times, 1, 1, 8, 0)
                tik_instance.vcadd(column_reduce_remainder,
                                   tensor_sec_reduce[i * half_mask_value + column_reduce_times],
                                   mid_result_ub[i * column_aligned + column_reduce_times * half_mask_value],
                                   1, 1, 1, 1, 0)
            tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce, row_nums, 1, 1, 8, 0)
        with tik_instance.else_scope():
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce, zeros_init, row_nums, 1, 8, 0)
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce[i * half_mask_value],
                                   mid_result_ub[i * column_aligned], column_reduce_times, 1, 1, 8, 0)
            tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce, row_nums, 1, 1, 8, 0)
    else:
        tensor_third_reduce = tik_instance.Tensor(dtype="float32",
                                                  shape=(Constant.BLOCK_SIZE, Constant.V_SIZE_BYTES),
                                                  name="tensor_third_reduce",
                                                  scope=tik.scope_ubuf)
        tensor_sec_reduce = tik_instance.Tensor(dtype="float32",
                                                shape=(Constant.BLOCK_SIZE, half_mask_value),
                                                name="tensor_sec_reduce",
                                                scope=tik.scope_ubuf)
        zeros_init = tik_instance.Scalar("float32")
        zeros_init.set_as(0)

        tik_instance.vector_dup(half_mask_value, tensor_third_reduce, zeros_init, row_nums * 4, 1, 8, 0)
        tik_instance.vector_dup(half_mask_value, tensor_sec_reduce, zeros_init, row_nums, 1, 8, 0)
        with tik_instance.if_scope(column_reduce_remainder != 0):
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value, tensor_third_reduce[i * Constant.V_SIZE_BYTES],
                                   mid_result_ub[i * column_aligned], column_reduce_times, 1, 1, 8, 0)
                tik_instance.vcadd(column_reduce_remainder,
                                   tensor_third_reduce[i * Constant.V_SIZE_BYTES + column_reduce_times],
                                   mid_result_ub[i * column_aligned + column_reduce_times * half_mask_value],
                                   1, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce[i * half_mask_value],
                                   tensor_third_reduce[i * Constant.V_SIZE_BYTES], 4, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce, row_nums, 1, 1, 8, 0)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value, tensor_third_reduce[i * Constant.V_SIZE_BYTES],
                                   mid_result_ub[i * column_aligned], column_reduce_times, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce[i * half_mask_value],
                                   tensor_third_reduce[i * Constant.V_SIZE_BYTES], 4, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce, row_nums, 1, 1, 8, 0)

    # if data_bool[i] < k, then the tensor_output[i] is true, else is false(represented by 0 and 1).
    data_k = tik_instance.Tensor(dtype="float32",
                                 shape=target_shape,
                                 name="data_k",
                                 scope=tik.scope_ubuf)
    tensor_bool = tik_instance.Tensor(dtype="float16",
                                      shape=target_shape,
                                      name="tensor_bool",
                                      scope=tik.scope_ubuf)

    repeat_times = row_nums // half_mask_value
    repeat_remainder = row_nums % half_mask_value

    with tik_instance.if_scope(repeat_remainder == 0):
        tik_instance.vector_dup(half_mask_value, data_k, k_conv_dtype, repeat_times, 1, 8, 0)
        tik_instance.vsub(half_mask_value, data_bool, data_k, data_bool, repeat_times, 1, 1, 1, 8, 8, 8)
        tik_instance.vconv(half_mask_value, '', tensor_bool, data_bool, repeat_times, 1, 1, 4, 8)
    with tik_instance.else_scope():
        with tik_instance.if_scope(repeat_times == 0):
            tik_instance.vector_dup(row_nums, data_k, k_conv_dtype, 1, 1, 1, 0)
            tik_instance.vsub(repeat_remainder, data_bool, data_k, data_bool, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vconv(repeat_remainder, '', tensor_bool, data_bool, 1, 1, 1, 8, 8)
        with tik_instance.else_scope():
            index = repeat_times * half_mask_value
            tik_instance.vector_dup(half_mask_value, data_k, k_conv_dtype, repeat_times, 1, 8, 0)
            tik_instance.vector_dup(repeat_remainder, data_k[index], k_conv_dtype, 1, 1, 1, 0)
            tik_instance.vsub(half_mask_value, data_bool, data_k, data_bool, repeat_times, 1, 1, 1, 8, 8, 8)
            tik_instance.vconv(half_mask_value, '', tensor_bool, data_bool, repeat_times, 1, 1, 4, 8)
            tik_instance.vsub(repeat_remainder, data_bool[index], data_k[index], data_bool[index], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vconv(repeat_remainder, '', tensor_bool[index], data_bool[index], 1, 1, 1, 4, 8)

    tensor_zeros = tik_instance.Tensor(dtype="float16",
                                       shape=(Constant.BLOCK_SIZE,),
                                       name="zeros",
                                       scope=tik.scope_ubuf)
    tensor_ones = tik_instance.Tensor(dtype="float16",
                                      shape=(Constant.BLOCK_SIZE,),
                                      name="ones",
                                      scope=tik.scope_ubuf)

    zeros_uint8 = tik_instance.Scalar("float16")
    zeros_uint8.set_as(0)
    tik_instance.vector_dup(Constant.BLOCK_SIZE, tensor_zeros, zeros_uint8, 1, 1, 1, 0)
    ones_uint8 = tik_instance.Scalar("float16")
    ones_uint8.set_as(1)
    tik_instance.vector_dup(Constant.BLOCK_SIZE, tensor_ones, ones_uint8, 1, 1, 1, 0)

    data_bool_ub = tik_instance.Tensor(dtype="float16",
                                       shape=target_shape,
                                       name="data_bool_ub",
                                       scope=tik.scope_ubuf)
    tensor_output_ub = tik_instance.Tensor(dtype=obj_gm.get_tensor_output_gm().dtype,
                                           shape=target_shape,
                                           name="tensor_output_ub",
                                           scope=tik.scope_ubuf)

    cmp_times = row_nums // Constant.BLOCK_SIZE
    cmp_rem = row_nums % Constant.BLOCK_SIZE
    src = tik_instance.Tensor(dtype="float16",
                              shape=(16,),
                              name="src",
                              scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, src, 0, 1, 1, 8)
    dst_ub = tik_instance.Tensor(dtype="int32",
                                 shape=(Constant.BLOCK_SIZE,),
                                 name="dst_ub",
                                 scope=tik.scope_ubuf)
    dst_ub1 = tik_instance.Tensor(dtype="float16",
                                  shape=(Constant.BLOCK_SIZE,),
                                  name="dst_ub1",
                                  scope=tik.scope_ubuf)

    with tik_instance.if_scope(cmp_times > 0):
        with tik_instance.for_range(0, cmp_times) as i:
            cmp_mask = tik_instance.vcmp_gt(Constant.BLOCK_SIZE,
                                            tensor_bool[i * Constant.BLOCK_SIZE], tensor_zeros, 1, 1)
            tik_instance.vsel(Constant.BLOCK_SIZE, 0, data_bool_ub[i * Constant.BLOCK_SIZE],
                              cmp_mask, tensor_ones, tensor_zeros, 1, 1, 1, 1)
            invalid_mask = calc_invalid_mask(tik_instance, target_ub[i * Constant.BLOCK_SIZE],
                                             Constant.BLOCK_SIZE, column, src, dst_ub, dst_ub1, 0)
            tik_instance.vsel(Constant.BLOCK_SIZE, 0, data_bool_ub[i * Constant.BLOCK_SIZE],
                              invalid_mask, tensor_zeros, data_bool_ub[i * Constant.BLOCK_SIZE], 1, 1, 1, 1)
            invalid_mask = calc_invalid_mask(tik_instance, target_ub[i * Constant.BLOCK_SIZE],
                                             Constant.BLOCK_SIZE, column, src, dst_ub, dst_ub1, 1)
            tik_instance.vsel(Constant.BLOCK_SIZE, 0, data_bool_ub[i * Constant.BLOCK_SIZE],
                              invalid_mask, tensor_zeros, data_bool_ub[i * Constant.BLOCK_SIZE], 1, 1, 1, 1)
            tik_instance.vconv(Constant.BLOCK_SIZE, '', tensor_output_ub[i * Constant.BLOCK_SIZE],
                               data_bool_ub[i * Constant.BLOCK_SIZE], 1, 1, 1, 8, 8)
    with tik_instance.if_scope(cmp_rem != 0):
        cmp_mask = tik_instance.vcmp_gt(cmp_rem, tensor_bool[cmp_times * Constant.BLOCK_SIZE], tensor_zeros, 1, 1)
        tik_instance.vsel(cmp_rem, 0, data_bool_ub[cmp_times * Constant.BLOCK_SIZE], cmp_mask,
                          tensor_ones, tensor_zeros, 1, 1, 1, 1)
        invalid_mask = calc_invalid_mask(tik_instance, target_ub[cmp_times * Constant.BLOCK_SIZE],
                                         cmp_rem, column, src, dst_ub, dst_ub1, 0)
        tik_instance.vsel(cmp_rem, 0, data_bool_ub[cmp_times * Constant.BLOCK_SIZE], invalid_mask,
                          tensor_zeros, data_bool_ub[cmp_times * Constant.BLOCK_SIZE], 1, 1, 1, 1)
        invalid_mask = calc_invalid_mask(tik_instance, target_ub[cmp_times * Constant.BLOCK_SIZE], cmp_rem,
                                         column, src, dst_ub, dst_ub1, 1)
        tik_instance.vsel(cmp_rem, 0, data_bool_ub[cmp_times * Constant.BLOCK_SIZE], invalid_mask,
                          tensor_zeros, data_bool_ub[cmp_times * Constant.BLOCK_SIZE], 1, 1, 1, 1)
        tik_instance.vconv(cmp_rem, '', tensor_output_ub[cmp_times * Constant.BLOCK_SIZE],
                           data_bool_ub[cmp_times * Constant.BLOCK_SIZE], 1, 1, 1, 8, 8)
    return tensor_output_ub


def _in_top_k_mul_core_v2(tik_instance, shape_info, obj_tiling, obj_gm, k, targets):
    """"
    the _in_top_k_mul_core_v2 function

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include keys(core_loop, mini_cloud_core_nums)
    obj_tiling: obj_tiling
                include keys(num_rows_scalar, num_cols_scalar, num_cores_scalar)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    target_dtype = targets.get("dtype")
    row = obj_tiling.get_rows_num()
    column = obj_tiling.get_cols_num()

    core_loop = shape_info.get("core_loop")
    mini_cloud_core_nums = shape_info.get("mini_cloud_core_nums")
    block_element = Constant.BLOCK_SIZE // Constant.BYTE_FLOAT32
    column_aligned = ((column + block_element - 1) // block_element * block_element)
    coexisting_tensor_num = 5
    max_tensor_size = Constant.UB_SIZE_BYTES // coexisting_tensor_num
    core_row_capicity = max_tensor_size // (column_aligned * Constant.BYTE_FLOAT32)

    split_core_nums = (row + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE
    split_rows_nums = Constant.BLOCK_SIZE
    split_remainder = row - (split_core_nums - 1) * Constant.BLOCK_SIZE
    non_tail_core = split_core_nums - 1

    row_nums = core_row_capicity
    single_core_times = (Constant.BLOCK_SIZE + core_row_capicity - 1) // core_row_capicity
    non_tail_block = single_core_times - 1
    row_remainder = Constant.BLOCK_SIZE - row_nums * non_tail_block

    last_single_core_times = (split_remainder + core_row_capicity - 1) // core_row_capicity
    last_non_tail_block = last_single_core_times - 1
    last_row_remainder = split_remainder - last_non_tail_block * core_row_capicity
    shape_info["targets_dtype"] = target_dtype
    with tik_instance.if_scope(split_core_nums <= mini_cloud_core_nums):
        with tik_instance.if_scope(core_loop < split_core_nums):
            outer_core_num = tik_instance.Scalar("int64")
            output_ub = tik_instance.Tensor(dtype=obj_gm.get_tensor_output_gm().dtype,
                                            shape=(Constant.BLOCK_SIZE,),
                                            name="output_ub",
                                            scope=tik.scope_ubuf)
            with tik_instance.if_scope(core_loop < non_tail_core):
                outer_core_num.set_as(split_rows_nums)
                with tik_instance.for_range(0, single_core_times) as outer_loop:
                    core_row_num = tik_instance.Scalar("int64")
                    with tik_instance.if_scope(outer_loop < non_tail_block):
                        core_row_num.set_as(row_nums)
                    with tik_instance.else_scope():
                        core_row_num.set_as(row_remainder)
                    shape_info["outer_loop"] = outer_loop
                    shape_info["index"] = core_loop * split_rows_nums + outer_loop * row_nums
                    shape_info["core_nums"] = single_core_times
                    shape_info["row_remainder"] = row_remainder
                    shape_info["row_nums"] = row_nums
                    shape_info["column"] = column
                    shape_info["column_aligned"] = column_aligned
                    shape_info["split_rows_nums"] = split_rows_nums
                    shape_info["max_tensor_size"] = max_tensor_size
                    with tik_instance.if_scope(column_aligned <= 4096):
                        tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm,
                                                                   k, Constant.BIG_DIM1_TYPE)
                        with tik_instance.for_range(0, core_row_num) as i:
                            output_ub[outer_loop * row_nums + i] = tensor_output_ub[i]
                    with tik_instance.else_scope():
                        tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm,
                                                                   k, Constant.OTHER_TYPE)
                        with tik_instance.for_range(0, core_row_num) as i:
                            output_ub[outer_loop * row_nums + i] = tensor_output_ub[i]
            with tik_instance.else_scope():
                outer_core_num.set_as(split_remainder)
                with tik_instance.for_range(0, last_single_core_times) as outer_loop:
                    core_row_num = tik_instance.Scalar("int64")
                    shape_info["outer_loop"] = outer_loop
                    shape_info["index"] = core_loop * split_rows_nums + outer_loop * row_nums
                    shape_info["core_nums"] = last_single_core_times
                    shape_info["row_remainder"] = last_row_remainder
                    shape_info["row_nums"] = row_nums
                    shape_info["column"] = column
                    shape_info["column_aligned"] = column_aligned
                    shape_info["split_rows_nums"] = split_rows_nums
                    shape_info["max_tensor_size"] = max_tensor_size
                    with tik_instance.if_scope(outer_loop < last_non_tail_block):
                        core_row_num.set_as(row_nums)
                    with tik_instance.else_scope():
                        core_row_num.set_as(last_row_remainder)
                    with tik_instance.if_scope(column_aligned <= 4096):
                        tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, 2)
                        with tik_instance.for_range(0, core_row_num) as i:
                            output_ub[outer_loop * row_nums + i] = tensor_output_ub[i]
                    with tik_instance.else_scope():
                        tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, 3)
                        with tik_instance.for_range(0, core_row_num) as i:
                            output_ub[outer_loop * row_nums + i] = tensor_output_ub[i]
            index = core_loop * Constant.BLOCK_SIZE
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], output_ub, 0, 1, 1, 0, 0)
    with tik_instance.else_scope():
        row_align = (row + Constant.BLOCK_SIZE - 1) // \
                    Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        core_split = tik_instance.Scalar(dtype="int64", name="core_split")
        index_base = tik_instance.Scalar(dtype="int64", name="index_base")
        split_time1 = tik_instance.Scalar(dtype="int64", name="split_time1")
        split_remainder1 = tik_instance.Scalar(dtype="int64", name="split_remainder1")
        output_ub = tik_instance.Tensor(dtype="uint8",
                                        shape=(Constant.BLOCK_SIZE,),
                                        name="output_ub",
                                        scope=tik.scope_ubuf)
        need_row_core1 = (row_align // mini_cloud_core_nums) // Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        need_row_core2 = (row_align // mini_cloud_core_nums + Constant.BLOCK_SIZE - 1) // \
                         Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        with tik_instance.if_scope(need_row_core1 == need_row_core2):
            core_split.set_as(0)
        with tik_instance.else_scope():
            core_split.set_as((need_row_core2 * mini_cloud_core_nums - row_align) // (need_row_core2 - need_row_core1))
        last_core = row - core_split * need_row_core1 - (mini_cloud_core_nums - core_split - 1) * need_row_core2

        with tik_instance.if_scope(core_loop < core_split):
            index_base.set_as(core_loop * need_row_core1)
            split_time1.set_as((need_row_core1 + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE)
            split_remainder1.set_as(need_row_core1 - (split_time1 - 1) * Constant.BLOCK_SIZE)
        with tik_instance.else_scope():
            index_base.set_as(need_row_core1 * core_split + (core_loop - core_split) * need_row_core2)
            with tik_instance.if_scope(core_loop < mini_cloud_core_nums - 1):
                split_time1.set_as((need_row_core2 + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE)
                split_remainder1.set_as(need_row_core2 - (split_time1 - 1) * Constant.BLOCK_SIZE)
            with tik_instance.else_scope():
                split_time1.set_as((last_core + Constant.BLOCK_SIZE) // Constant.BLOCK_SIZE)
                split_remainder1.set_as(last_core - (split_time1 - 1) * Constant.BLOCK_SIZE)

        with tik_instance.for_range(0, split_time1) as split_loop:
            split_time2 = tik_instance.Scalar(dtype="int64", name="split_time2")
            split_remainder2 = tik_instance.Scalar(dtype="int64", name="split_remainder2")
            with tik_instance.if_scope(split_loop < split_time1 - 1):
                split_time2.set_as((Constant.BLOCK_SIZE + core_row_capicity - 1) // core_row_capicity)
                split_remainder2.set_as(Constant.BLOCK_SIZE - (split_time2 - 1) * core_row_capicity)
            with tik_instance.else_scope():
                split_time2.set_as((split_remainder1 + core_row_capicity - 1) // core_row_capicity)
                split_remainder2.set_as(split_remainder1 - (split_time2 - 1) * core_row_capicity)
            with tik_instance.for_range(0, split_time2) as outer_loop:
                core_row_num = tik_instance.Scalar(dtype="int64", name="core_row_num")
                shape_info["outer_loop"] = outer_loop
                shape_info["index"] = index_base + Constant.BLOCK_SIZE * split_loop + \
                                      core_row_capicity * outer_loop
                shape_info["core_nums"] = split_time2
                shape_info["row_remainder"] = split_remainder2
                shape_info["row_nums"] = core_row_capicity
                shape_info["column"] = column
                shape_info["column_aligned"] = column_aligned
                shape_info["max_tensor_size"] = max_tensor_size
                num = tik_instance.Scalar(dtype="int64", name="num")
                num.set_as(split_time2 - 1)
                with tik_instance.if_scope(outer_loop < num):
                    core_row_num.set_as(core_row_capicity)
                with tik_instance.else_scope():
                    core_row_num.set_as(split_remainder2)
                with tik_instance.if_scope(column_aligned <= 4096):
                    tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, 2)
                    with tik_instance.for_range(0, core_row_num) as i:
                        output_ub[outer_loop * core_row_capicity + i] = tensor_output_ub[i]
                with tik_instance.else_scope():
                    tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, 3)
                    with tik_instance.for_range(0, core_row_num) as i:
                        output_ub[outer_loop * core_row_capicity + i] = tensor_output_ub[i]
            index = index_base + Constant.BLOCK_SIZE * split_loop
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], output_ub, 0, 1, 1, 0, 0)


def _in_top_k_mul_core(tik_instance, shape_info, obj_tiling, obj_gm, k, targets):
    """"
    the _in_top_k_mul_core function

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include keys(core_loop, mini_cloud_core_nums)
    obj_tiling: obj_tiling
                include keys(num_rows_scalar, num_cols_scalar, num_cores_scalar)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    element_bytes = 4
    coexisting_tensor_num = 5
    row = obj_tiling.get_rows_num()
    column = obj_tiling.get_cols_num()
    block_element = Constant.BLOCK_SIZE // element_bytes
    column_aligned = ((column + block_element - 1) // block_element * block_element)
    max_tensor_size = Constant.UB_SIZE_BYTES // coexisting_tensor_num
    split_rows_nums = 0
    mini_cloud_core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    core_row_capicity = max_tensor_size // (column_aligned * element_bytes)
    core_loop = shape_info.get("core_loop")

    row_split_num = (row + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE
    row_nums = tik_instance.Scalar(dtype="int64", name="row_nums")

    shape_info["targets_dtype"] = targets.get("dtype")
    with tik_instance.if_scope(row_split_num <= mini_cloud_core_nums):
        row_nums.set_as(Constant.BLOCK_SIZE)
        core_nums = (row + row_nums - 1) // row_nums
        tail_block = core_nums - 1
        row_remainder = row - row_nums * tail_block
    with tik_instance.else_scope():
        core_nums = mini_cloud_core_nums
        row_align = (row + Constant.BLOCK_SIZE - 1) // \
                    Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        row_nums.set_as(row_align // mini_cloud_core_nums // \
                       Constant.BLOCK_SIZE * Constant.BLOCK_SIZE)
        tail_block = core_nums - 1
        row_remainder = row - row_nums * tail_block

    # copy predictions to ub from gm,
    # if the last dimension is not divided by 32 bytes, just aligned.
    shape_info["core_loop"] = core_loop
    with tik_instance.if_scope(row_split_num <= mini_cloud_core_nums):
        shape_info["index"] = core_loop * row_nums
        shape_info["outer_loop"] = core_loop
        shape_info["core_nums"] = core_nums
        shape_info["row_remainder"] = row_remainder
        shape_info["row_nums"] = row_nums
        shape_info["column"] = column
        shape_info["column_aligned"] = column_aligned
        shape_info["max_tensor_size"] = max_tensor_size
        length = tik_instance.Scalar("int64")
        with tik_instance.if_scope(core_loop < tail_block):
            length.set_as(row_nums + Constant.BLOCK_SIZE - 1)
        with tik_instance.else_scope():
            length.set_as(row_remainder + Constant.BLOCK_SIZE - 1)
        with tik_instance.if_scope(column_aligned <= 64):
            tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, Constant.BIG_DIM0_TYPE)
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[core_loop * row_nums], tensor_output_ub,
                                   0, 1, length // Constant.BLOCK_SIZE, 0, 0)
        with tik_instance.else_scope():
            tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, Constant.BIG_DIM1_TYPE)
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[core_loop * row_nums], tensor_output_ub,
                                   0, 1, length // Constant.BLOCK_SIZE, 0, 0)
    with tik_instance.else_scope():
        core_split = tik_instance.Scalar(dtype="int64", name="core_split")
        index_base = tik_instance.Scalar(dtype="int64", name="index_base")
        split_time = tik_instance.Scalar(dtype="int64", name="split_time")
        core_row_remainder = tik_instance.Scalar(dtype="int64", name="core_row_remainder")
        row_core_line = tik_instance.Scalar(dtype="int64", name="row_core_line")
        need_row_core1 = (row_align // mini_cloud_core_nums) // \
                         Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        need_row_core2 = (row_align // mini_cloud_core_nums + Constant.BLOCK_SIZE - 1) // \
                         Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        with tik_instance.if_scope(need_row_core1 == need_row_core2):
            core_split.set_as(0)
        with tik_instance.else_scope():
            core_split.set_as((need_row_core2 * mini_cloud_core_nums - row_align) // (need_row_core2 - need_row_core1))
        row_remainder = row - core_split * need_row_core1 - (mini_cloud_core_nums - core_split - 1) * need_row_core2
        row_core_line.set_as(core_row_capicity // Constant.BLOCK_SIZE * Constant.BLOCK_SIZE)
        with tik_instance.if_scope(row_core_line > need_row_core2):
            row_core_line.set_as(need_row_core2)
        with tik_instance.if_scope(core_loop < core_split):
            index_base.set_as(core_loop * need_row_core1)
            split_time.set_as((need_row_core1 + row_core_line - 1) // row_core_line)
            core_row_remainder.set_as(need_row_core1 - (split_time - 1) * row_core_line)
        with tik_instance.else_scope():
            index_base.set_as(need_row_core1 * core_split + (core_loop - core_split) * need_row_core2)
            with tik_instance.if_scope(core_loop < core_nums - 1):
                split_time.set_as((need_row_core2 + row_core_line - 1) // row_core_line)
                core_row_remainder.set_as(need_row_core2 - (split_time - 1) * row_core_line)
            with tik_instance.else_scope():
                split_time.set_as((row_remainder + row_core_line - 1) // row_core_line)
                core_row_remainder.set_as(row_remainder - (split_time - 1) * row_core_line)

        with tik_instance.for_range(0, split_time) as outer_loop:
            shape_info["index"] = index_base + outer_loop * row_core_line
            shape_info["outer_loop"] = outer_loop
            shape_info["core_nums"] = split_time
            shape_info["row_remainder"] = core_row_remainder
            shape_info["row_nums"] = row_core_line
            shape_info["column"] = column
            shape_info["column_aligned"] = column_aligned
            shape_info["split_rows_nums"] = split_rows_nums
            shape_info["max_tensor_size"] = max_tensor_size
            index = index_base + outer_loop * row_core_line
            length = tik_instance.Scalar("int64")
            with tik_instance.if_scope(outer_loop < split_time - 1):
                length.set_as(row_core_line + Constant.BLOCK_SIZE - 1)
            with tik_instance.else_scope():
                length.set_as(core_row_remainder + Constant.BLOCK_SIZE - 1)
            with tik_instance.if_scope(column_aligned <= 64):
                tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, 1)
                tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], tensor_output_ub, 0, 1,
                                       length // Constant.BLOCK_SIZE, 0, 0)
            with tik_instance.else_scope():
                tensor_output_ub = _in_top_k_inter_process(tik_instance, shape_info, obj_gm, k, 2)
                tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], tensor_output_ub, 0, 1,
                                       length // Constant.BLOCK_SIZE, 0, 0)


def _in_top_k_column_process(tik_instance, shape_info, obj_gm, k_conv_dtype):
    """"
    the _in_top_k_column_process function
    the process of _in_top_k_tiling_column

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include keys(core_loop, mini_cloud_core_nums, index, outer_loop,
                             core_nums, row_remainder, row_nums, column, column_aligned,
                             split_rows_nums, max_tensor_size)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    ----------
    """
    index = shape_info.get("index")
    inner_loop = shape_info.get("inner_loop")
    split_column_nums = shape_info.get("split_column_nums")
    column = shape_info.get("column")
    column_num = shape_info.get("column_num")
    column_size = shape_info.get("column_size")
    max_tensor_size = shape_info.get("max_tensor_size")
    scalar_value = shape_info.get("scalar_value")
    half_mask_value = 64
    mask_value = 128
    carry = 2
    block_element = Constant.BLOCK_SIZE // Constant.BYTE_FLOAT32
    element_bytes = 4
    prediction_dtype = "float32"

    # step 1: do vector_dup in data_ub, in this situation, row_nums is 1.
    column_aligned = (column + block_element - 1) // block_element * block_element
    prediction_tensor_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                               shape=(1, max_tensor_size // element_bytes),
                                               name="prediction_tensor_ub",
                                               scope=tik.scope_ubuf)
    index = index * column
    tik_instance.data_move(prediction_tensor_ub, obj_gm.get_predictions_gm()[index + inner_loop * column_size], 0, 1,
                           column_num // block_element, 0, 0)

    # dirty data set as FP_MIN, for example, A[:, 1:15] is target data,
    # then set A[:, 15:16] as FP_MIN.
    with tik_instance.if_scope(inner_loop == split_column_nums - 1):
        reg_data = tik_instance.Scalar(prediction_dtype)
        reg_data.set_as(Constant.FP_MIN)
        index = column - inner_loop * column_size
        with tik_instance.for_range(0, column_aligned - column) as j:
            prediction_tensor_ub[0, index + j] = reg_data

    data_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                  shape=(1, max_tensor_size // element_bytes),
                                  name="data_ub",
                                  scope=tik.scope_ubuf)
    # set the number of repeat.
    column_reduce_times = column_num // half_mask_value
    column_reduce_remainder = column_num % half_mask_value

    if Constant.CHECK_INF:
        with tik_instance.if_scope(column_reduce_times > 0):
            tik_instance.vaxpy(half_mask_value,
                               prediction_tensor_ub, prediction_tensor_ub,
                               0.0, column_reduce_times, 1, 1, 8, 8)
        with tik_instance.if_scope(column_reduce_remainder != 0):
            tik_instance.vaxpy(column_reduce_remainder,
                               prediction_tensor_ub[column_reduce_times * half_mask_value],
                               prediction_tensor_ub[column_reduce_times * half_mask_value],
                               0.0, 1, 1, 1, 8, 8)

    with tik_instance.if_scope(column_reduce_times >= 1):
        tik_instance.vector_dup(half_mask_value, data_ub, scalar_value, column_reduce_times, 1, 8, 0)
    with tik_instance.if_scope(column_reduce_remainder != 0):
        tik_instance.vector_dup(column_reduce_remainder, data_ub[column_reduce_times * half_mask_value], scalar_value,
                                1, 1, 1, 0)

    if tbe_platform.api_check_support("tik.vcmp_gt", "float32"):
        data_zeros = tik_instance.Tensor(dtype="float32",
                                         shape=(half_mask_value, 1),
                                         name="data_zeros",
                                         scope=tik.scope_ubuf)
        tik_instance.vector_dup(half_mask_value, data_zeros, 0, 1, 1, 1, 0)
        data_ones = tik_instance.Tensor(dtype="float32",
                                        shape=(half_mask_value, 1),
                                        name="data_ones",
                                        scope=tik.scope_ubuf)
        tik_instance.vector_dup(half_mask_value, data_ones, 1, 1, 1, 1, 0)

        data_sign = tik_instance.Tensor(dtype="float32",
                                        shape=(max_tensor_size // element_bytes,),
                                        name="data_sign",
                                        scope=tik.scope_ubuf)
        repeat_times = column_num // half_mask_value
        tail_mask = column_num % half_mask_value
        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vector_dup(half_mask_value, data_sign, 0, repeat_times, 1, 8)
        with tik_instance.if_scope(tail_mask != 0):
            tik_instance.vector_dup(tail_mask, data_sign[repeat_times * half_mask_value], 0, 1, 1, 8)

        if Constant.CHECK_INF:
            data_cmpmask = tik_instance.Tensor(dtype="uint64",
                                               shape=(4, ),
                                               name="data_cmpmask",
                                               scope=tik.scope_ubuf)
            tik_instance.vector_dup(8, data_cmpmask.reinterpret_cast_to("uint32"), 0, 1, 1, 1, 0)
            scalar_nan_check_mask = tik_instance.Scalar("uint64", "scalar_nan_check_mask")
            scalar_tail_mask = tik_instance.Scalar("uint64", "scalar_tail_mask", init_value=1)
            # bit shift: scalar_tail_mask = 1 << column_reduce_remainder
            with tik_instance.for_range(0, column_reduce_remainder) as i:
                scalar_tail_mask = scalar_tail_mask * 2
            scalar_tail_mask = ~ (scalar_tail_mask - 1)
            tmp_reduce_data_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                               shape=(half_mask_value, ),
                                               name="tmp_reduce_data_ub",
                                               scope=tik.scope_ubuf)

            with tik_instance.if_scope(column_reduce_times > 0):
                tik_instance.vector_dup(half_mask_value, tmp_reduce_data_ub, 0.0, 1, 1, 8)
                with tik_instance.for_range(0, column_reduce_times) as j:
                    tik_instance.vmax(half_mask_value,
                        tmp_reduce_data_ub,
                        prediction_tensor_ub[half_mask_value * j],
                        tmp_reduce_data_ub,
                        1, 1, 1, 1, 8, 8, 8)
                with tik_instance.if_scope(column_reduce_remainder != 0):
                    tik_instance.vmax(column_reduce_remainder,
                        tmp_reduce_data_ub,
                        prediction_tensor_ub[column_reduce_times * half_mask_value],
                        tmp_reduce_data_ub,
                        1, 1, 1, 1, 8, 8, 8)

                nan_check = tik_instance.vcmp_eq(half_mask_value, tmp_reduce_data_ub, tmp_reduce_data_ub, 1, 1)
                tik_instance.mov_cmpmask_to_tensor(data_cmpmask, nan_check)
                scalar_nan_check_mask.set_as(data_cmpmask[0])

                with tik_instance.if_scope(scalar_nan_check_mask == 0xffffffffffffffff):
                    with tik_instance.for_range(0, column_reduce_times) as j:
                        srcmask = tik_instance.vcmp_gt(half_mask_value,
                                                       prediction_tensor_ub[half_mask_value * j], data_ub,
                                                       1, 1)
                        tik_instance.vsel(half_mask_value, 0,
                                          data_sign[half_mask_value * j],
                                          srcmask, data_ones, data_zeros,
                                          1, 1, 1, 1)
                    with tik_instance.if_scope(column_reduce_remainder != 0):
                        srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                    prediction_tensor_ub[column_reduce_times * half_mask_value],
                                                    data_ub[column_reduce_times * half_mask_value], 1, 1)
                        tik_instance.vsel(column_reduce_remainder, 0,
                                          data_sign[column_reduce_times * half_mask_value],
                                          srcmask, data_ones, data_zeros,
                                          1, 1, 1, 1)
                with tik_instance.else_scope():
                    data_sign[0].set_as(k_conv_dtype)
            with tik_instance.else_scope():
                nan_check = tik_instance.vcmp_eq(column_reduce_remainder,
                                                 prediction_tensor_ub, prediction_tensor_ub,
                                                 1, 1)
                tik_instance.mov_cmpmask_to_tensor(data_cmpmask, nan_check)
                scalar_nan_check_mask.set_as(data_cmpmask[0])
                scalar_nan_check_mask = scalar_nan_check_mask | scalar_tail_mask

                with tik_instance.if_scope(scalar_nan_check_mask == 0xffffffffffffffff):
                    srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                   prediction_tensor_ub, data_ub,
                                                   1, 1)
                    tik_instance.vsel(column_reduce_remainder, 0,
                                      data_sign, srcmask, data_ones, data_zeros,
                                      1, 1, 1, 1)
                with tik_instance.else_scope():
                    data_sign[0].set_as(k_conv_dtype)
        else:
            with tik_instance.if_scope(column_reduce_times > 0):
                with tik_instance.for_range(0, column_reduce_times) as j:
                    srcmask = tik_instance.vcmp_gt(half_mask_value,
                                                   prediction_tensor_ub[half_mask_value * j], data_ub,
                                                   1, 1)
                    tik_instance.vsel(half_mask_value, 0,
                                      data_sign[half_mask_value * j],
                                      srcmask, data_ones, data_zeros,
                                      1, 1, 1, 1)
                with tik_instance.if_scope(column_reduce_remainder != 0):
                    srcmask = tik_instance.vcmp_gt(column_reduce_remainder,
                                                prediction_tensor_ub[column_reduce_times * half_mask_value],
                                                data_ub[column_reduce_times * half_mask_value], 1, 1)
                    tik_instance.vsel(column_reduce_remainder, 0, data_sign[column_reduce_times * half_mask_value],
                                    srcmask, data_ones, data_zeros, 1, 1, 1, 1)
            with tik_instance.else_scope():
                srcmask = tik_instance.vcmp_gt(column_reduce_remainder, prediction_tensor_ub, data_ub, 1, 1)
                tik_instance.vsel(column_reduce_remainder, 0, data_sign, srcmask, data_ones, data_zeros, 1, 1, 1, 1)
        mid_result_ub = data_sign

    else:
        # step 2: prediction_tensor_ub subtract data_ub.
        half_repeat_times = column_num // half_mask_value
        half_tail_mask = column_num % half_mask_value
        half_sub = tik_instance.Tensor(dtype="float16",
                                       shape=(max_tensor_size // element_bytes * carry,),
                                       name="half_sub",
                                       scope=tik.scope_ubuf)

        with tik_instance.if_scope(half_repeat_times != 0):
            tik_instance.vsub(half_mask_value, prediction_tensor_ub, prediction_tensor_ub, data_ub, half_repeat_times,
                              1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(half_tail_mask != 0):
            index = half_repeat_times * half_mask_value
            tik_instance.vsub(half_tail_mask, prediction_tensor_ub[index], prediction_tensor_ub[index], data_ub[index],
                              1, 1, 1, 1, 8, 8, 8)

        repeat_times = (carry * column_num) // mask_value
        tail_mask = (carry * column_num) % mask_value

        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vector_dup(mask_value, half_sub, 0, repeat_times, 1, 8)
        with tik_instance.if_scope(tail_mask != 0):
            tik_instance.vector_dup(tail_mask, half_sub[repeat_times * mask_value], 0, 1, 1, 8)

        with tik_instance.if_scope(column_reduce_times > 0):
            tik_instance.vconv(half_mask_value, '', half_sub, prediction_tensor_ub, column_reduce_times, 1, 1, 4, 8)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                index = half_mask_value * column_reduce_times
                tik_instance.vconv(column_reduce_remainder, '', half_sub[index], prediction_tensor_ub[index],
                                   1, 1, 1, 4, 8)
        with tik_instance.else_scope():
            tik_instance.vconv(column_reduce_remainder, '', half_sub, prediction_tensor_ub, 1, 1, 1, 4, 8)

        # step 3: if half_sub[i, j] > 0, then the accoding element data_sign[i, j]
        # set 1, else set 0.
        column_reduce_times = column_num // mask_value
        column_reduce_remainder = column_num % mask_value

        data_sign = tik_instance.Tensor(dtype="float16",
                                        shape=(max_tensor_size // element_bytes * carry,),
                                        name="data_sign",
                                        scope=tik.scope_ubuf)
        data_zeros = tik_instance.Tensor(dtype="float16",
                                         shape=(mask_value, 1),
                                         name="data_zeros",
                                         scope=tik.scope_ubuf)
        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vector_dup(mask_value, data_sign, 0, repeat_times, 1, 8)
        with tik_instance.if_scope(tail_mask != 0):
            tik_instance.vector_dup(tail_mask, data_sign[repeat_times * mask_value], 0, 1, 1, 8)

        tik_instance.vector_dup(mask_value, data_zeros, 0, 1, 1, 1, 0)
        data_ones = tik_instance.Tensor(dtype="float16",
                                        shape=(mask_value, 1),
                                        name="data_ones",
                                        scope=tik.scope_ubuf)
        tik_instance.vector_dup(mask_value, data_ones, 1, 1, 1, 1, 0)

        with tik_instance.if_scope(column_reduce_times > 0):
            with tik_instance.for_range(0, column_reduce_times) as j:
                srcmask = tik_instance.vcmp_gt(mask_value, half_sub[mask_value * j], data_zeros, 1, 1)
                tik_instance.vsel(mask_value, 0, data_sign[mask_value * j], srcmask, data_ones, data_zeros, 1, 1, 1, 1)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                srcmask = tik_instance.vcmp_gt(column_reduce_remainder, half_sub[column_reduce_times * mask_value],
                                               data_zeros, 1, 1)
                tik_instance.vsel(column_reduce_remainder, 0, data_sign[column_reduce_times * mask_value], srcmask,
                                  data_ones, data_zeros, 1, 1, 1, 1)
        with tik_instance.else_scope():
            srcmask = tik_instance.vcmp_gt(column_reduce_remainder, half_sub, data_zeros, 1, 1)
            tik_instance.vsel(column_reduce_remainder, 0, data_sign, srcmask, data_ones, data_zeros, 1, 1, 1, 1)

        column_reduce_times = column_num // half_mask_value
        column_reduce_remainder = column_num % half_mask_value

        with tik_instance.if_scope(column_reduce_times > 0):
            tik_instance.vconv(half_mask_value, '', prediction_tensor_ub, data_sign, column_reduce_times, 1, 1, 8, 4)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                tik_instance.vconv(column_reduce_remainder, '',
                                   prediction_tensor_ub[column_reduce_times * half_mask_value],
                                   data_sign[column_reduce_times * half_mask_value], 1, 1, 1, 8, 4)
        with tik_instance.else_scope():
            tik_instance.vconv(column_reduce_remainder, '', prediction_tensor_ub, data_sign, 1, 1, 1, 8, 4)

        mid_result_ub = prediction_tensor_ub

    with tik_instance.if_scope(column_reduce_remainder != 0):
        reduce_mask = column_reduce_times + 1
    with tik_instance.else_scope():
        reduce_mask = column_reduce_times

    data_bool = tik_instance.Tensor(dtype="float32",
                                    shape=(1,),
                                    name="data_bool",
                                    scope=tik.scope_ubuf)
    with tik_instance.if_scope(reduce_mask == 1):
        with tik_instance.if_scope(column_reduce_remainder != 0):
            tik_instance.vcadd(column_reduce_remainder, data_bool, mid_result_ub, 1, 1, 1, 1, 0)
        with tik_instance.else_scope():
            tik_instance.vcadd(half_mask_value, data_bool, mid_result_ub, 1, 1, 1, 1, 0)
    with tik_instance.else_scope():
        with tik_instance.if_scope(reduce_mask <= half_mask_value):
            tensor_sec_reduce = tik_instance.Tensor(dtype="float32",
                                                    shape=(1, half_mask_value),
                                                    name="tensor_sec_reduce",
                                                    scope=tik.scope_ubuf)
            zeros_init = tik_instance.Scalar("float32")
            zeros_init.set_as(0)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                tik_instance.vector_dup(half_mask_value, tensor_sec_reduce, zeros_init, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce, mid_result_ub, column_reduce_times,
                                   1, 1, 8, 0)
                tik_instance.vcadd(column_reduce_remainder, tensor_sec_reduce[column_reduce_times],
                                   mid_result_ub[column_reduce_times * half_mask_value], 1, 1, 1, 1, 0)
                tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce, 1, 1, 1, 8, 0)
            with tik_instance.else_scope():
                tik_instance.vector_dup(half_mask_value, tensor_sec_reduce, zeros_init, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce, mid_result_ub, column_reduce_times, 1,
                                   1, 8, 0)
                tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce, 1, 1, 1, 8, 0)
        with tik_instance.else_scope():
            tensor_third_reduce = tik_instance.Tensor(dtype="float32",
                                                      shape=(1, Constant.V_SIZE_BYTES),
                                                      name="tensor_third_reduce",
                                                      scope=tik.scope_ubuf)
            tensor_sec_reduce = tik_instance.Tensor(dtype="float32",
                                                    shape=(1, half_mask_value),
                                                    name="tensor_sec_reduce",
                                                    scope=tik.scope_ubuf)
            zeros_init = tik_instance.Scalar("float32")
            zeros_init.set_as(0)
            tik_instance.vector_dup(half_mask_value, tensor_third_reduce, zeros_init, 4, 1, 8, 0)
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce, zeros_init, 1, 1, 8, 0)
            with tik_instance.if_scope(column_reduce_remainder != 0):
                tik_instance.vcadd(half_mask_value, tensor_third_reduce, mid_result_ub, column_reduce_times, 1,
                                   1, 8, 0)
                tik_instance.vcadd(column_reduce_remainder, tensor_third_reduce[column_reduce_times],
                                   mid_result_ub[column_reduce_times * half_mask_value], 1, 1, 1, 1, 0)
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce, tensor_third_reduce, 4, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce, 1, 1, 1, 8, 0)
            with tik_instance.else_scope():
                tik_instance.vcadd(half_mask_value, tensor_third_reduce, mid_result_ub, column_reduce_times, 1,
                                   1, 8, 0)
                tik_instance.vcadd(half_mask_value, tensor_sec_reduce, tensor_third_reduce, 4, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce, 1, 1, 1, 8, 0)

    return data_bool


def _in_top_k_column_inner_loop(tik_instance, shape_info, obj_gm, k):
    """"
    the _in_top_k_column_inner_loop function

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include keys(core_loop, mini_cloud_core_nums, index, single_core_times,
                             split_column_nums, column_size, column, column_num,
                             last_column_size, max_tensor_size)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    index = shape_info.get("index")
    column = shape_info.get("column")
    column_size = shape_info.get("column_size")
    last_column_size = shape_info.get("last_column_size")
    split_column_nums = shape_info.get("split_column_nums")
    prediction_dtype = "float32"
    target_dtype = shape_info.get("targets_dtype")
    element_bytes = 4
    block_element = Constant.BLOCK_SIZE // element_bytes

    target_ub = tik_instance.Tensor(dtype=target_dtype,
                                    shape=(block_element,),
                                    name="target_ub",
                                    scope=tik.scope_ubuf)
    data_temp_ub = tik_instance.Tensor(dtype=prediction_dtype,
                                       shape=(block_element,),
                                       name="data_temp_ub",
                                       scope=tik.scope_ubuf)
    tensor_output_ub_temp = tik_instance.Tensor(dtype=obj_gm.get_tensor_output_gm().dtype,
                                                shape=(Constant.BLOCK_SIZE,),
                                                name="tensor_output_ub_temp",
                                                scope=tik.scope_ubuf)
    k_conv_dtype = tik_instance.Scalar(prediction_dtype)
    k_conv_dtype.set_as(k)
    # copy target_tensor to UB.
    # just need 1 element each time, but read 8 elements
    tik_instance.data_move(target_ub, obj_gm.get_targets_gm()[index], 0, 1, 1, 0, 0)
    scalar_target = tik_instance.Scalar(target_dtype, "scalar_target")
    scalar_target.set_as(target_ub[0])
    with tik_instance.if_scope(tik.any(scalar_target < 0, scalar_target >= column)):
        scalar_target.set_as(0)
    tik_instance.data_move(data_temp_ub, obj_gm.get_predictions_gm()[index * column + scalar_target], 0, 1, 1, 0, 0)

    # get the value according to the target of each row.
    scalar_value = tik_instance.Scalar(prediction_dtype)
    scalar_value.set_as(data_temp_ub[0])
    shape_info["scalar_value"] = scalar_value
    bool_src = tik_instance.Tensor(dtype=prediction_dtype,
                                   shape=(1,),
                                   name="bool_src",
                                   scope=tik.scope_ubuf)
    tik_instance.vector_dup(1, bool_src, 0, 1, 1, 1)
    # record the number of element which is larger than target element.
    bool_sum = tik_instance.Tensor(dtype=prediction_dtype,
                                   shape=(1,),
                                   name="bool_sum",
                                   scope=tik.scope_ubuf)
    tik_instance.vector_dup(1, bool_sum, 0, 1, 1, 1)

    with tik_instance.for_range(0, split_column_nums) as inner_loop:
        with tik_instance.if_scope(inner_loop < split_column_nums - 1):
            shape_info["inner_loop"] = inner_loop
            shape_info["column_num"] = column_size
            data_bool = _in_top_k_column_process(tik_instance, shape_info, obj_gm, k_conv_dtype)
            tik_instance.vadd(1, bool_sum, bool_src, data_bool, 1, 1, 1, 1, 8, 8, 8)
            sum_scalar = tik_instance.Scalar(prediction_dtype, name="sum_scalar")
            sum_scalar.set_as(0)
            tik_instance.vadds(1, bool_src, bool_sum, sum_scalar, 1, 1, 1, 8, 8, 0)
        with tik_instance.else_scope():
            shape_info["inner_loop"] = inner_loop
            shape_info["column_num"] = last_column_size
            data_bool = _in_top_k_column_process(tik_instance, shape_info, obj_gm, k_conv_dtype)
            tik_instance.vadd(1, bool_sum, bool_src, data_bool, 1, 1, 1, 1, 8, 8, 8)
            sum_scalar = tik_instance.Scalar(prediction_dtype, name="sum_scalar")
            sum_scalar.set_as(0)
            tik_instance.vadds(1, bool_src, bool_sum, sum_scalar, 1, 1, 1, 8, 8, 0)

    tensor_k = tik_instance.Tensor(dtype=prediction_dtype,
                                   shape=(8,),
                                   name="tensor_k",
                                   scope=tik.scope_ubuf)
    tensor_zeros = tik_instance.Tensor(dtype="float16",
                                       shape=(8,),
                                       name="tensor_zeros",
                                       scope=tik.scope_ubuf)
    tensor_ones = tik_instance.Tensor(dtype="float16",
                                      shape=(8,),
                                      name="tensor_ones",
                                      scope=tik.scope_ubuf)

    tik_instance.vector_dup(1, tensor_k, k_conv_dtype, 1, 1, 1, 0)

    tensor_sub_float = tik_instance.Tensor(dtype=prediction_dtype,
                                           shape=(1,),
                                           name="tensor_sub_float",
                                           scope=tik.scope_ubuf)
    tensor_sub_half = tik_instance.Tensor(dtype="float16",
                                          shape=(1,),
                                          name="tensor_sub_half",
                                          scope=tik.scope_ubuf)
    tik_instance.vsub(1, tensor_sub_float, bool_sum, tensor_k, 1, 1, 1, 1, 8, 8, 8)
    tik_instance.vconv(1, '', tensor_sub_half, tensor_sub_float, 1, 1, 1, 8, 8)

    zeros_half = tik_instance.Scalar("float16")
    zeros_half.set_as(0)
    tik_instance.vector_dup(1, tensor_zeros, zeros_half, 1, 1, 1, 0)
    ones_half = tik_instance.Scalar("float16")
    ones_half.set_as(1)
    tik_instance.vector_dup(1, tensor_ones, ones_half, 1, 1, 1, 0)

    src = tik_instance.Tensor(dtype="float16",
                              shape=(16,),
                              name="src",
                              scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, src, 0, 1, 1, 8)
    dst_ub = tik_instance.Tensor(dtype="int32",
                                 shape=(Constant.BLOCK_SIZE,),
                                 name="dst_ub",
                                 scope=tik.scope_ubuf)
    dst_ub1 = tik_instance.Tensor(dtype="float16",
                                  shape=(Constant.BLOCK_SIZE,),
                                  name="dst_ub1",
                                  scope=tik.scope_ubuf)
    data_bool_ub = tik_instance.Tensor(dtype="float16",
                                       shape=(1,),
                                       name="data_bool_ub",
                                       scope=tik.scope_ubuf)
    cmp_mask = tik_instance.vcmp_lt(1, tensor_sub_half, tensor_zeros, 1, 1)
    tik_instance.vsel(1, 0, data_bool_ub, cmp_mask, tensor_ones, tensor_zeros, 1, 1, 1, 1)
    invalid_mask = calc_invalid_mask(tik_instance, target_ub, block_element, column, src, dst_ub,
                                     dst_ub1, 0)
    tik_instance.vsel(1, 0, data_bool_ub, invalid_mask, tensor_zeros, data_bool_ub, 1, 1, 1, 1)
    invalid_mask = calc_invalid_mask(tik_instance, target_ub, block_element, column, src, dst_ub,
                                     dst_ub1, 1)
    tik_instance.vsel(1, 0, data_bool_ub, invalid_mask, tensor_zeros, data_bool_ub, 1, 1, 1, 1)
    tik_instance.vconv(1, '', tensor_output_ub_temp, data_bool_ub, 1, 1, 1, 8, 8)

    return tensor_output_ub_temp


def _in_top_k_tiling_column(tik_instance, shape_info, obj_tiling, obj_gm, k, targets):
    """"
    the _in_top_k_tiling_column function

    Parameters
    ----------
    tik_instance: tik_instance
    shape_info: dict
                include keys(core_loop, mini_cloud_core_nums)
    obj_tiling: obj_tiling
                include keys(num_rows_scalar, num_cols_scalar, num_cores_scalar)
    obj_gm: obj_gm
            include keys(prediction_gm, target_gm, tensor_output_gm, tiling_gm)
    k: the k value of top k
    ----------
    """
    column = obj_tiling.get_cols_num()
    mini_cloud_core_nums = shape_info.get("mini_cloud_core_nums")
    core_loop = shape_info.get("core_loop")
    block_element = Constant.BLOCK_SIZE // Constant.BYTE_FLOAT32
    column_aligned = ((column + block_element - 1) // block_element * block_element)
    coexisting_tensor_num = 5
    max_tensor_size = Constant.UB_SIZE_BYTES // coexisting_tensor_num // block_element * block_element - block_element

    split_column_nums = (column_aligned * Constant.BYTE_FLOAT32 + max_tensor_size - 1) // max_tensor_size
    column_num_temp = column_aligned // split_column_nums
    column_num = (column_num_temp + block_element - 1) // block_element * block_element + block_element
    last_column_size = column_aligned - (column_num * (split_column_nums - 1))
    row = obj_tiling.get_rows_num()

    single_core_times = tik_instance.Scalar(dtype="int64", name="single_core_times")
    split_row_times = tik_instance.Scalar(dtype="int64", name="split_row_times")
    row_remainder = tik_instance.Scalar(dtype="int64", name="row_remainder")
    with tik_instance.if_scope(row <= Constant.BLOCK_SIZE):
        split_row_times.set_as(1)
        single_core_times.set_as(row)
        row_remainder.set_as(row)
    with tik_instance.else_scope():
        split_row_times.set_as((row + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE)
        single_core_times.set_as(Constant.BLOCK_SIZE)
        row_remainder.set_as(row - (split_row_times - 1) * Constant.BLOCK_SIZE)

    move_data_count = tik_instance.Scalar(dtype="int64", name="move_data_count")
    move_data_count.set_as(0)
    shape_info['targets_dtype'] = targets.get("dtype")
    with tik_instance.if_scope(split_row_times <= mini_cloud_core_nums):
        output_ub = tik_instance.Tensor(dtype=obj_gm.get_tensor_output_gm().dtype,
                                        shape=(Constant.BLOCK_SIZE,),
                                        name="output_ub",
                                        scope=tik.scope_ubuf)
        with tik_instance.if_scope(core_loop < split_row_times - 1):
            with tik_instance.for_range(0, single_core_times) as outer_loop:
                shape_info["index"] = core_loop * single_core_times + outer_loop
                shape_info["single_core_times"] = single_core_times
                shape_info["split_column_nums"] = split_column_nums
                shape_info["column_size"] = column_num
                shape_info["column"] = column
                shape_info["column_num"] = last_column_size
                shape_info["last_column_size"] = last_column_size
                shape_info["max_tensor_size"] = max_tensor_size
                tensor_output_temp_ub = _in_top_k_column_inner_loop(tik_instance, shape_info, obj_gm, k)
                output_ub[outer_loop] = tensor_output_temp_ub[0]
            index = core_loop * single_core_times
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], output_ub, 0, 1, 1, 1, 1)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, row_remainder) as outer_loop:
                shape_info["index"] = core_loop * single_core_times + outer_loop
                shape_info["single_core_times"] = single_core_times
                shape_info["split_column_nums"] = split_column_nums
                shape_info["column_size"] = column_num
                shape_info["column"] = column
                shape_info["column_num"] = last_column_size
                shape_info["last_column_size"] = last_column_size
                shape_info["max_tensor_size"] = max_tensor_size
                tensor_output_temp_ub = _in_top_k_column_inner_loop(tik_instance, shape_info, obj_gm, k)
                output_ub[outer_loop] = tensor_output_temp_ub[0]
            index = core_loop * single_core_times
            tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], output_ub, 0, 1, 1, 1, 1)
    with tik_instance.else_scope():
        row_align = (row + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        core_split = tik_instance.Scalar(dtype="int64", name="core_split")
        index_base = tik_instance.Scalar(dtype="int64", name="index_base")
        split_time1 = tik_instance.Scalar(dtype="int64", name="split_time1")
        need_row_core1 = (row_align // mini_cloud_core_nums) // Constant.BLOCK_SIZE * Constant.BLOCK_SIZE
        need_row_core2 = (row_align // mini_cloud_core_nums + Constant.BLOCK_SIZE - 1) // \
                         Constant.BLOCK_SIZE * Constant.BLOCK_SIZE

        with tik_instance.if_scope(need_row_core1 == need_row_core2):
            core_split.set_as(0)
        with tik_instance.else_scope():
            core_split.set_as((need_row_core2 * mini_cloud_core_nums - row_align) // (need_row_core2 - need_row_core1))
        last_core = row - core_split * need_row_core1 - (mini_cloud_core_nums - core_split - 1) * need_row_core2

        with tik_instance.if_scope(core_loop < core_split):
            index_base.set_as(core_loop * need_row_core1)
            split_time1.set_as((need_row_core1 + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE)
            row_remainder.set_as(need_row_core1 - (split_time1 -  1) * Constant.BLOCK_SIZE)
        with tik_instance.else_scope():
            index_base.set_as(need_row_core1 * core_split + (core_loop - core_split) * need_row_core2)
            with tik_instance.if_scope(core_loop < mini_cloud_core_nums - 1):
                split_time1.set_as((need_row_core2 + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE)
                row_remainder.set_as(need_row_core2 - (split_time1 - 1) * Constant.BLOCK_SIZE)
            with tik_instance.else_scope():
                split_time1.set_as((last_core + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE)
                row_remainder.set_as(last_core - (split_time1 - 1) * Constant.BLOCK_SIZE)

        output_ub = tik_instance.Tensor(dtype=obj_gm.get_tensor_output_gm().dtype,
                                        shape=(Constant.BLOCK_SIZE,),
                                        name="output_ub",
                                        scope=tik.scope_ubuf)
        with tik_instance.for_range(0, split_time1) as split_loop:
            with tik_instance.if_scope(split_loop < split_time1 - 1):
                with tik_instance.for_range(0, Constant.BLOCK_SIZE) as outer_loop:
                    shape_info["index"] = index_base + split_loop * Constant.BLOCK_SIZE + outer_loop
                    shape_info["single_core_times"] = single_core_times
                    shape_info["split_column_nums"] = split_column_nums
                    shape_info["column_size"] = column_num
                    shape_info["column"] = column
                    shape_info["column_num"] = last_column_size
                    shape_info["last_column_size"] = last_column_size
                    shape_info["max_tensor_size"] = max_tensor_size
                    tensor_output_temp_ub = _in_top_k_column_inner_loop(tik_instance, shape_info, obj_gm, k)
                    output_ub[outer_loop] = tensor_output_temp_ub[0]
                index = index_base + split_loop * Constant.BLOCK_SIZE
                tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], output_ub, 0, 1, 1, 1, 1)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, row_remainder) as outer_loop:
                    shape_info["index"] = index_base + split_loop * Constant.BLOCK_SIZE + outer_loop
                    shape_info["single_core_times"] = single_core_times
                    shape_info["split_column_nums"] = split_column_nums
                    shape_info["column_size"] = column_num
                    shape_info["column"] = column
                    shape_info["column_num"] = last_column_size
                    shape_info["last_column_size"] = last_column_size
                    shape_info["max_tensor_size"] = max_tensor_size
                    tensor_output_temp_ub = _in_top_k_column_inner_loop(tik_instance, shape_info, obj_gm, k)
                    output_ub[outer_loop] = tensor_output_temp_ub[0]
                index = index_base + split_loop * Constant.BLOCK_SIZE
                tik_instance.data_move(obj_gm.get_tensor_output_gm()[index], output_ub, 0, 1, 1, 1, 1)
