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
scatter_elements
"""
from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 48
    BYTES_PER_BLOCK = 32
    MAX_INT32 = 2**31 - 1
    MAX_DIM = 8
    TILING_MODE_SAMESHAPE = 0
    TILING_MODE_DIFFSHAPE = 1
    TILING_MODE_LASTAXIS_SAME_SHAPE = 2
    TILING_MODE_LASTAXIS_DIFF_SHAPE = 3
    DTYPE_BYTES_DICT = {"float16": 2, "float32": 4, "int64": 8, "int32": 4, "uint8": 1,
                        "int8": 1, "bool": 1, "bfloat16": 2}


# 'pylint: disable=too-many-locals, too-many-arguments
def check_supported(data, indices, updates, result, axis = 0, reduction = "none", kernel_name="scatter_elements"):
    """
        check the op support situation.
    """
    data_shape = data.get("shape")
    if int(-1) in data_shape or int(-2) in data_shape:
        return "Unknown"
    return False


class Processor:
    def process_offset(self, offset, indices_ub, indices_block_offset, index_i):
        pass


class ProcessorNormal(Processor):
    def __init__(self, tik_instance, shape_indices_ub, shape_data_ub, dims_data, dtype_indices, dims_indices,
                 shape_acc_indices, axis):
        self.tik_instance = tik_instance
        self.dims_indices = dims_indices
        self.dims_data = dims_data
        self.shape_data_ub = shape_data_ub
        self.dtype_indices = dtype_indices
        self.axis = axis
        self.indices_value = tik_instance.Tensor(dtype_indices, [dims_indices], name="indices_value",
                                                 scope=tik.scope_ubuf)
        self.shape_acc_cur = tik_instance.Scalar(dtype=dtype_indices, name="shape_acc_cur")
        self.shape_indices_cur = tik_instance.Scalar(dtype=dtype_indices, name="shape_indices_cur")
        self.shape_acc = tik_instance.Tensor(dtype_indices, [dims_indices], name="shape_acc", scope=tik.scope_ubuf)
        self.shape_indices_ub = shape_indices_ub
        shape_indices_temp = tik_instance.Scalar(dtype=dtype_indices, name="shape_indices_temp",
                                                 init_value=shape_indices_ub[0])
        self.shape_acc[0].set_as(shape_acc_indices // shape_indices_temp)
        with self.tik_instance.for_range(1, self.dims_indices) as dim:
            self.shape_acc_cur.set_as(self.shape_acc[dim - 1])
            self.shape_indices_cur.set_as(shape_indices_ub[dim])
            self.shape_acc[dim].set_as(self.shape_acc_cur // self.shape_indices_cur)

        self.shape_indices_next = self.tik_instance.Scalar(dtype=self.dtype_indices, name="shape_indices_next")
        self.indices_value_next = self.tik_instance.Scalar(dtype=self.dtype_indices, name="indices_value_next")
        self.indices_offset = self.tik_instance.Scalar(dtype=self.dtype_indices, name="indices_offset")
        self.index_dim = self.tik_instance.Scalar(dtype=self.dtype_indices, name="index_dim")

    def process_offset(self, offset, indices_ub, indices_block_offset, index_i):
        # calculate offset of current element of indices in data
        self.indices_offset.set_as(indices_block_offset + index_i)
        with self.tik_instance.for_range(0, self.dims_indices) as dim:
            self.shape_acc_cur.set_as(self.shape_acc[dim])
            self.shape_indices_cur.set_as(self.shape_indices_ub[dim])
            self.indices_value[dim].set_as(self.indices_offset // self.shape_acc_cur % self.shape_indices_cur)
        self.index_dim.set_as(indices_ub[index_i])
        self.indices_value[self.axis].set_as(self.index_dim)
        offset.set_as(self.indices_value[0])
        with self.tik_instance.for_range(0, self.dims_data - 1) as j:
            self.shape_indices_next.set_as(self.shape_data_ub[j + 1])
            self.indices_value_next.set_as(self.indices_value[j + 1])
            offset.set_as(offset * self.shape_indices_next + self.indices_value_next)


class ProcessorAxis(Processor):
    def __init__(self, tik_instance, dtype_indices, params_row_indices, params_axis_indices, params_row_data,
                 params_axis_data):
        self.tik_instance = tik_instance
        self.dtype_indices = dtype_indices
        self.params_row_indices = params_row_indices
        self.params_axis_indices = params_axis_indices
        self.params_axis_data = params_axis_data
        self.params_row_data = params_row_data

        self.indices_offset = self.tik_instance.Scalar(dtype=self.dtype_indices, name="indices_offset")
        self.tail_row = self.tik_instance.Scalar(dtype=self.dtype_indices, name="tail_row")
        self.loop_pre = self.tik_instance.Scalar(dtype=self.dtype_indices, name="loop_pre")
        self.indices_value_i = self.tik_instance.Scalar(dtype=self.dtype_indices, name="indices_value_i")

    def process_offset(self, offset, indices_ub, indices_block_offset, index_i):
        # calculate offset of current element of indices in data
        self.indices_offset.set_as(indices_block_offset + index_i)
        self.tail_row.set_as(self.indices_offset % self.params_row_indices)
        self.loop_pre.set_as(self.indices_offset // self.params_axis_indices)
        self.indices_value_i.set_as(indices_ub[index_i])
        offset.set_as(self.loop_pre * self.params_axis_data + self.indices_value_i * self.params_row_data +
                      self.tail_row)


# 'pylint: disable=too-many-locals, too-many-arguments
class ScatterElements:
    """ class for scatter_elements """

    # 'pylint: disable=too-many-arguments
    def __init__(self, data, indices, updates, result, axis, reduction, kernel_name):
        """__init__"""
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.reduction = reduction
        self.tiling_param_dtype = 'int32'
        self.axis = self.tik_instance.Scalar(self.tiling_param_dtype, name='axis')
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_param_dtype, (Constant.TILING_ARG_NUM,), name='tiling_gm',
                                                  scope=tik.scope_gm)

        self.dtype_data = data.get("dtype").lower()
        self.dtype_indices = indices.get("dtype").lower()
        self.dtype_updates = updates.get("dtype").lower()
        self.dtype_out = result.get("dtype").lower()

        self.indices_each_block = Constant.BYTES_PER_BLOCK // Constant.DTYPE_BYTES_DICT.get(self.dtype_indices, 8)
        self.updates_each_block = Constant.BYTES_PER_BLOCK // Constant.DTYPE_BYTES_DICT.get(self.dtype_data, 4)

        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)) // Constant.BYTES_PER_BLOCK * \
                             Constant.BYTES_PER_BLOCK

        self.data_gm = self.tik_instance.Tensor(self.dtype_data, [Constant.MAX_INT32], name="data_gm",
                                                scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.dtype_indices, [Constant.MAX_INT32], name="indices_gm",
                                                   scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.dtype_updates, [Constant.MAX_INT32], name="updates_gm",
                                                   scope=tik.scope_gm)
        self.result_gm = self.tik_instance.Tensor(self.dtype_out, [Constant.MAX_INT32], name="result_gm",
                                                  scope=tik.scope_gm)

        self.dims_indices = self.tik_instance.Scalar(self.tiling_param_dtype, name='dims_indices')
        self.dims_data = self.tik_instance.Scalar(self.tiling_param_dtype, name='dims_data')
        self.shape_acc_data = self.tik_instance.Scalar(self.tiling_param_dtype, name='shape_acc_data')
        self.shape_acc_indices = self.tik_instance.Scalar(self.tiling_param_dtype, name='shape_acc_indices')
        self.data_block = self.tik_instance.Scalar(self.tiling_param_dtype, name='data_block')
        self.repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='repeat')
        self.used_aicore_num = self.tik_instance.Scalar(self.tiling_param_dtype, name='used_aicore_num')
        self.batch_num_per_aicore = self.tik_instance.Scalar(self.tiling_param_dtype, name='batch_num_per_aicore')
        self.batch_tail = self.tik_instance.Scalar(self.tiling_param_dtype, name='batch_tail')
        self.indices_block = self.tik_instance.Scalar(self.tiling_param_dtype, name='indices_block')
        self.indices_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='indices_repeat')
        self.rounds_indices = self.tik_instance.Scalar(self.tiling_param_dtype, name='rounds_indices')
        self.tail_indices = self.tik_instance.Scalar(self.tiling_param_dtype, name='tail_indices')
        self.tail_indices_block = self.tik_instance.Scalar(self.tiling_param_dtype, name='tail_indices_block')
        self.tail_indices_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='tail_indices_repeat')
        self.updates_block = self.tik_instance.Scalar(self.tiling_param_dtype, name='updates_block')
        self.updates_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='updates_repeat')
        self.tail_updates_block = self.tik_instance.Scalar(self.tiling_param_dtype, name='tail_updates_block')
        self.tail_updates_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='tail_updates_repeat')
        self.tiling_mode = self.tik_instance.Scalar(self.tiling_param_dtype, name='tiling_mode')
        self.params_row_data = self.tik_instance.Scalar(self.tiling_param_dtype, name='params_row_data')
        self.params_row_indices = self.tik_instance.Scalar(self.tiling_param_dtype, name='params_row_indices')
        self.params_axis_data = self.tik_instance.Scalar(self.tiling_param_dtype, name='params_axis_data')
        self.params_axis_indices = self.tik_instance.Scalar(self.tiling_param_dtype, name='params_axis_indices')

        self.shape_indices_ub = self.tik_instance.Tensor(self.tiling_param_dtype, [Constant.MAX_DIM],
                                                         name="shape_indices_ub", scope=tik.scope_ubuf)
        self.shape_data_ub = self.tik_instance.Tensor(self.tiling_param_dtype, [Constant.MAX_DIM], name="shape_data_ub",
                                                      scope=tik.scope_ubuf)

        self.rounds = self.tik_instance.Scalar(self.tiling_param_dtype, name='rounds')
        self.data_tail_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='data_tail_repeat')
        self.repeat_per_core = self.tik_instance.Scalar(self.tiling_param_dtype, name='repeat_per_core')
        self.rounds_tail = self.tik_instance.Scalar(self.tiling_param_dtype, name='rounds_tail')

        self.processor = Processor()
        self.indices_block_offset = self.tik_instance.Scalar(self.tiling_param_dtype, name='indices_block_offset')
        self.indices_ub = None
        self.updates_ub = None
        self.indices_updates_size = self.tik_instance.Scalar(self.tiling_param_dtype, name='indices_updates_size')

        if self.dtype_data == "bfloat16":
            self.updates_temp_bf16_ub = self.tik_instance.Tensor("bfloat16", (1,), name="updates_temp_bf16_ub",
                                                                 scope=tik.scope_ubuf)
            self.updates_temp_fp32_ub = self.tik_instance.Tensor("float32", (1,), name="updates_temp_fp32_ub",
                                                                 scope=tik.scope_ubuf)
            self.data_temp_bf16_ub = self.tik_instance.Tensor("bfloat16", (1,), name="data_temp_bf16_ub",
                                                              scope=tik.scope_ubuf)
            self.data_temp_fp32_ub = self.tik_instance.Tensor("float32", (1,), name="data_temp_fp32_ub",
                                                              scope=tik.scope_ubuf)
            self.data_temp_fp32 = self.tik_instance.Scalar("float32", name="data_temp_fp32")
            self.updates_temp_fp32 = self.tik_instance.Scalar("float32", name="updates_temp_fp32")

    def get_tiling_args(self):
        """
        get_tiling_args
        """
        tiling_ub = self.tik_instance.Tensor(self.tiling_param_dtype, (Constant.TILING_ARG_NUM,),
                                             name='tiling_ub',
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                    Constant.TILING_ARG_NUM // 8, 0, 0) # 8 for int32

        self.dims_indices.set_as(tiling_ub[0])
        self.dims_data.set_as(tiling_ub[1])
        self.shape_acc_data.set_as(tiling_ub[2])
        self.shape_acc_indices.set_as(tiling_ub[3])

        with self.tik_instance.for_range(0, self.dims_indices) as i:
            # get shape_indices_ub from tiling_ub[4] to tiling_ub[4 + dims_indices], will reach tiling_ub[11] at most
            self.shape_indices_ub[i].set_as(tiling_ub[i + 4])
            # get shape_data_ub from tiling_ub[12] to tiling_ub[12 + dims_indices], will reach tiling_ub[19] at most
            self.shape_data_ub[i].set_as(tiling_ub[i + 12])
        with self.tik_instance.for_range(self.dims_indices, Constant.MAX_DIM) as i:
            # other elements of shape_indices_ub and shape_data_ub are set as 0
            # they will not be used in any function, setting to 0 is just for security
            self.shape_indices_ub[i].set_as(0)
            self.shape_data_ub[i].set_as(0)

        self.data_block.set_as(tiling_ub[20])
        self.repeat.set_as(tiling_ub[21])
        self.used_aicore_num.set_as(tiling_ub[22])
        self.batch_num_per_aicore.set_as(tiling_ub[23])
        self.batch_tail.set_as(tiling_ub[24])
        self.indices_block.set_as(tiling_ub[25])
        self.indices_repeat.set_as(tiling_ub[26])
        self.rounds_indices.set_as(tiling_ub[27])
        self.tail_indices.set_as(tiling_ub[28])
        self.tail_indices_block.set_as(tiling_ub[29])
        self.tail_indices_repeat.set_as(tiling_ub[30])
        self.updates_block.set_as(tiling_ub[31])
        self.updates_repeat.set_as(tiling_ub[32])
        self.tail_updates_block.set_as(tiling_ub[33])
        self.tail_updates_repeat.set_as(tiling_ub[34])
        self.tiling_mode.set_as(tiling_ub[35])
        self.params_row_data.set_as(tiling_ub[36])
        self.params_row_indices.set_as(tiling_ub[37])
        self.params_axis_data.set_as(tiling_ub[38])
        self.params_axis_indices.set_as(tiling_ub[39])
        self.axis.set_as(tiling_ub[40])
        self.rounds.set_as(tiling_ub[41])
        self.data_tail_repeat.set_as(tiling_ub[42])
        self.repeat_per_core.set_as(tiling_ub[43])
        self.rounds_tail.set_as(tiling_ub[44])

        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, [self.indices_block], name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(self.dtype_updates, [self.updates_block], name="updates_ub",
                                                   scope=tik.scope_ubuf)

    def update_data(self, data_ub, updates_ub, data_offset, updates_offset):
        """
            update_data
            update data_ub by values in updates_ub
        """
        if (self.reduction == "none"):
            data_ub[data_offset].set_as(updates_ub[updates_offset])
        elif (self.reduction == "add"):
            if self.dtype_data == "float16":
                data_temp_fp16 = self.tik_instance.Scalar(dtype=self.dtype_data, name="data_temp_fp16",
                                                          init_value=data_ub[data_offset])
                data_temp = self.tik_instance.Scalar(dtype="float32", name="data_temp")
                self.tik_instance.scalar_conv('', data_temp, data_temp_fp16)
                updates_temp_fp16 = self.tik_instance.Scalar(dtype=self.dtype_data,
                                                             name="updates_temp_fp16",
                                                             init_value=updates_ub[updates_offset])
                updates_temp = self.tik_instance.Scalar(dtype="float32", name="updates_temp")
                self.tik_instance.scalar_conv('', updates_temp, updates_temp_fp16)
                data_ub_temp = self.tik_instance.Scalar(dtype="float32", name="updates_temp",
                                                        init_value=(data_temp + updates_temp))
                data_ub_temp_fp16 = self.tik_instance.Scalar(dtype=self.dtype_data,
                                                             name="data_ub_temp_fp16")
                self.tik_instance.scalar_conv('', data_ub_temp_fp16, data_ub_temp)
                data_ub[data_offset].set_as(data_ub_temp_fp16)
            elif self.dtype_data == "bfloat16":
                self.data_temp_bf16_ub[0].set_as(data_ub[data_offset]) # no
                self.updates_temp_bf16_ub[0].set_as(updates_ub[updates_offset])

                self.tik_instance.vec_conv(1, "", self.data_temp_fp32_ub, self.data_temp_bf16_ub, 1, 0, 0)
                self.tik_instance.vec_conv(1, "", self.updates_temp_fp32_ub, self.updates_temp_bf16_ub, 1, 0, 0) # y

                self.data_temp_fp32.set_as(self.data_temp_fp32_ub[0])
                self.updates_temp_fp32.set_as(self.updates_temp_fp32_ub[0])
                self.data_temp_fp32_ub[0].set_as(self.updates_temp_fp32 + self.data_temp_fp32) # y
                self.tik_instance.vec_conv(1, "round", self.data_temp_bf16_ub, self.data_temp_fp32_ub, 1, 0, 0) # y
                data_ub[data_offset].set_as(self.data_temp_bf16_ub[0])
            else:
                data_temp = self.tik_instance.Scalar(dtype=self.dtype_data, name="data_temp",
                                                    init_value=data_ub[data_offset])
                updates_temp = self.tik_instance.Scalar(dtype=self.dtype_data, name="updates_temp",
                                                        init_value=updates_ub[updates_offset])
                data_ub[data_offset].set_as(data_temp + updates_temp)

    # 'pylint: disable=too-many-arguments, huawei-too-many-arguments
    def update_data_from_updates(self, data_ub, task_id, cur_data_block, offset, indices_repeat_cur, updates_repeat_cur,
                                 indices_len_cur):
        """
            update_data_from_updates
        """
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[self.indices_block_offset], 0, 1,
                                    indices_repeat_cur, 0, 0)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm[self.indices_block_offset], 0, 1,
                                    updates_repeat_cur, 0, 0)
        with self.tik_instance.for_range(0, indices_len_cur) as index_i:
            self.processor.process_offset(offset, self.indices_ub, self.indices_block_offset, index_i)
            # Check whether the element of indices maps to the current data block according to
            # the offset above. If so, update data.
            offset.set_as(offset - task_id * self.data_block)
            with self.tik_instance.if_scope(tik.all(offset >= 0, offset < cur_data_block)):
                self.update_data(data_ub, self.updates_ub, offset, index_i)

    def compute_core(self, task_id):
        """
            compute_core
        """
        cur_data_block = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="cur_data_block",
                                                  init_value=self.data_block)
        data_tail_block = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_tail_block",
                                                   init_value=self.data_block / self.repeat * self.data_tail_repeat)
        cur_data_repeat = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="cur_data_repeat",
                                                   init_value=self.repeat)
        with self.tik_instance.if_scope(task_id == self.rounds - 1): # data tail block
            cur_data_block.set_as(data_tail_block)
            cur_data_repeat.set_as(self.data_tail_repeat)
        data_ub = self.tik_instance.Tensor(self.dtype_data, [cur_data_block], name="data_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub, self.data_gm[task_id * self.data_block], 0, 1, cur_data_repeat, 0, 0)

        offset = self.tik_instance.Scalar(dtype=self.dtype_indices, name="offset")
        with self.tik_instance.for_range(0, self.rounds_indices) as i:
            self.indices_block_offset.set_as(self.indices_block * i)
            self.update_data_from_updates(data_ub, task_id, cur_data_block, offset, self.indices_repeat,
                                          self.updates_repeat, self.indices_block)

        with self.tik_instance.if_scope(self.tail_indices != 0):
            self.indices_block_offset.set_as(self.indices_block * self.rounds_indices)
            self.update_data_from_updates(data_ub, task_id, cur_data_block, offset, self.tail_indices_repeat,
                                          self.tail_updates_repeat, self.tail_indices)

        with self.tik_instance.if_scope(task_id != self.rounds - 1):
            self.tik_instance.data_move(self.result_gm[task_id * self.data_block], data_ub, 0, 1, self.repeat, 0, 0)
        with self.tik_instance.else_scope():  # data tail block
            self.tik_instance.data_move(self.result_gm[task_id * self.data_block], data_ub, 0, 1,
                                        self.data_tail_repeat, 0, 0)

    def cal_indices_col_offset(self, task_id, indices_col, is_within_indices):
        """
            cal_indices_col_offset
            calculate the current indices column with each task_id
        """
        length_acc_cur = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="length_acc_cur",
                                                  init_value=self.shape_data_ub[0])
        shape_data_cur = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="shape_data_cur")
        with self.tik_instance.for_range(1, self.dims_data - 1) as dim:  # self.dims_indices >= 2 for diffshape
            shape_data_cur.set_as(self.shape_data_ub[dim])
            length_acc_cur.set_as(length_acc_cur * shape_data_cur)
        shape_indices_next = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="shape_indices_next")
        indices_col_next = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_col_next")
        length_ordinate = self.tik_instance.Tensor(self.dtype_indices, [self.dims_data - 1], name="length_ordinate",
                                                   scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.dims_data - 1) as dim:
            shape_data_cur.set_as(self.shape_data_ub[dim])
            length_acc_cur.set_as(length_acc_cur / shape_data_cur)
            length_ordinate[dim].set_as(task_id // length_acc_cur % shape_data_cur)
        indices_col.set_as(length_ordinate[0])
        length_acc_cur.set_as(self.shape_indices_ub[0])
        with self.tik_instance.if_scope(indices_col > (length_acc_cur - 1)):
            is_within_indices.set_as(0)
        with self.tik_instance.for_range(0, self.dims_indices - 2) as j:
            shape_indices_next.set_as(self.shape_indices_ub[j + 1])
            indices_col_next.set_as(length_ordinate[j + 1])
            with self.tik_instance.if_scope(indices_col_next > (shape_indices_next - 1)):
                is_within_indices.set_as(0)
            indices_col.set_as(indices_col * shape_indices_next + indices_col_next)

    def compute_core_last_axis(self, task_id):
        """
            compute_core
            for last axis cases
            sameshape + diffshape version
        """
        indices_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_width",
                                                 init_value=self.shape_indices_ub[self.axis])
        data_offset = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_offset")

        indices_col = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_col", init_value=task_id)
        is_within_indices = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="is_within_indices",
                                                     init_value=1)
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_LASTAXIS_DIFF_SHAPE):
            self.cal_indices_col_offset(task_id, indices_col, is_within_indices)

        data_ub = self.tik_instance.Tensor(self.dtype_data, [self.data_block], name="data_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub, self.data_gm[task_id * self.data_block], 0, 1, self.repeat, 0, 0)

        with self.tik_instance.if_scope(is_within_indices == 1):
            self.indices_block_offset.set_as(indices_col * indices_width)
            with self.tik_instance.for_range(0, self.rounds_indices) as i:
                self.tik_instance.data_move(self.indices_ub,
                                            self.indices_gm[self.indices_block_offset + i * self.indices_block],
                                            0, 1, self.indices_repeat, 0, 0)
                self.tik_instance.data_move(self.updates_ub,
                                            self.updates_gm[self.indices_block_offset + i * self.indices_block],
                                            0, 1, self.updates_repeat, 0, 0)
                with self.tik_instance.for_range(0, self.indices_block) as offset:
                    data_offset.set_as(self.indices_ub[offset])
                    self.update_data(data_ub, self.updates_ub, data_offset, offset)
            with self.tik_instance.if_scope(self.tail_indices != 0):
                self.indices_block_offset.set_as(indices_col * indices_width + self.rounds_indices * self.indices_block)
                self.tik_instance.data_move(self.indices_ub, self.indices_gm[self.indices_block_offset],
                                            0, 1, self.tail_indices_repeat, 0, 0)
                self.tik_instance.data_move(self.updates_ub, self.updates_gm[self.indices_block_offset],
                                            0, 1, self.tail_updates_repeat, 0, 0)
                with self.tik_instance.for_range(0, self.tail_indices) as offset:
                    data_offset.set_as(self.indices_ub[offset])
                    self.update_data(data_ub, self.updates_ub, data_offset, offset)

        self.tik_instance.data_move(self.result_gm[task_id * self.data_block], data_ub, 0, 1, self.repeat, 0, 0)

    def compute_core_last_axis_unaligned_big(self, task_id, data_repeat_per_block):
        """
            compute_core
            for last axis cases with unaligned data
            sameshape version
        """
        data_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_width",
                                              init_value=self.shape_data_ub[self.axis])
        block = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="block",
                                         init_value=self.data_block / self.repeat)
        indices_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_width",
                                                 init_value=self.shape_indices_ub[self.axis])
        data_offset = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_offset")
        data_ub = self.tik_instance.Tensor(self.dtype_data, [data_width * data_repeat_per_block], name="data_ub",
                                           scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub, self.data_gm[task_id * data_width * self.repeat_per_core], 0, 1,
                                    (data_width * data_repeat_per_block + block - 1) // block, 0, 0)

        with self.tik_instance.for_range(0, data_repeat_per_block) as i:
            self.indices_block_offset.set_as((task_id * self.repeat_per_core + i) * indices_width)
            with self.tik_instance.for_range(0, self.rounds_indices) as j:
                self.tik_instance.data_move(self.indices_ub,
                                            self.indices_gm[self.indices_block_offset + j * self.indices_block], 0, 1,
                                            self.indices_repeat, 0, 0)
                self.tik_instance.data_move(self.updates_ub,
                                            self.updates_gm[self.indices_block_offset + j * self.indices_block], 0, 1,
                                            self.updates_repeat, 0, 0)
                with self.tik_instance.for_range(0, self.indices_block) as offset:
                    data_offset.set_as(self.indices_ub[offset])
                    data_offset.set_as(data_offset + i * data_width)
                    self.update_data(data_ub, self.updates_ub, data_offset, offset)
            with self.tik_instance.if_scope(self.tail_indices != 0):
                self.indices_block_offset.set_as((task_id * self.repeat_per_core + i) * indices_width +
                                                 self.rounds_indices * self.indices_block)
                self.tik_instance.data_move(self.indices_ub, self.indices_gm[self.indices_block_offset], 0, 1,
                                            self.tail_indices_repeat, 0, 0)
                self.tik_instance.data_move(self.updates_ub, self.updates_gm[self.indices_block_offset], 0, 1,
                                            self.tail_updates_repeat, 0, 0)
                with self.tik_instance.for_range(0, self.tail_indices) as offset:
                    data_offset.set_as(self.indices_ub[offset])
                    data_offset.set_as(data_offset + i * data_width)
                    self.update_data(data_ub, self.updates_ub, data_offset, offset)

        self.tik_instance.data_move(self.result_gm[task_id * data_width * self.repeat_per_core], data_ub, 0, 1,
                                    (data_width * data_repeat_per_block + block - 1) // block, 0, 0)

    def compute_core_last_axis_unaligned(self, task_id, data_repeat_per_block):
        """
            compute_core
            for last axis cases with unaligned data
            sameshape version
        """
        data_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_width",
                                            init_value=self.shape_data_ub[self.axis])
        block = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="block",
                                        init_value=self.data_block / self.repeat)
        indices_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_width",
                                                init_value=self.shape_indices_ub[self.axis])
        data_offset = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_offset")
        data_ub = self.tik_instance.Tensor(self.dtype_data, [data_width * data_repeat_per_block], name="data_ub",
                                        scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub, self.data_gm[task_id * data_width * self.repeat_per_core], 0, 1,
                                    (data_width * data_repeat_per_block + block - 1) // block, 0, 0)
        base_offset = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="base_offset")

        indices_repeat_times = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_repeat_times",
                                                init_value=self.indices_updates_size // self.indices_each_block)
        self.tik_instance.data_move(self.indices_ub,
                                    self.indices_gm[task_id * self.repeat_per_core * indices_width], 0, 1,
                                    indices_repeat_times, 0, 0)
        updates_repeat_times = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_repeat_times",
                                                init_value=self.indices_updates_size // self.updates_each_block)
        self.tik_instance.data_move(self.updates_ub,
                                    self.updates_gm[task_id * self.repeat_per_core * indices_width], 0, 1,
                                    updates_repeat_times, 0, 0)

        with self.tik_instance.for_range(0, data_repeat_per_block) as i:
            with self.tik_instance.for_range(0, self.rounds_indices) as j:
                base_offset = i * indices_width + j * self.indices_block
                with self.tik_instance.for_range(0, self.indices_block) as offset:
                    data_offset.set_as(self.indices_ub[offset + base_offset])
                    data_offset.set_as(data_offset + i * data_width)
                    self.update_data(data_ub, self.updates_ub, data_offset, offset + base_offset)
            with self.tik_instance.if_scope(self.tail_indices != 0):
                base_offset = i * indices_width + self.rounds_indices * self.indices_block
                with self.tik_instance.for_range(0, self.tail_indices) as offset:
                    data_offset.set_as(self.indices_ub[offset + base_offset])
                    data_offset.set_as(data_offset + i * data_width)
                    self.update_data(data_ub, self.updates_ub, data_offset, offset + base_offset)

        self.tik_instance.data_move(self.result_gm[task_id * data_width * self.repeat_per_core], data_ub, 0, 1,
                                    (data_width * data_repeat_per_block + block - 1) // block, 0, 0)

    def compute_core_last_axis_pre(self, task):
        indices_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_width",
                                                init_value=self.shape_indices_ub[self.axis])
        with self.tik_instance.if_scope(self.shape_data_ub[self.axis] == self.data_block):
            self.compute_core_last_axis(task)
        with self.tik_instance.else_scope(): # unaligned cases
            with self.tik_instance.if_scope(tik.all(self.rounds_tail != 0, (task + 1) == self.rounds)):
                self.indices_updates_size.set_as(self.rounds_tail * indices_width
                                                 + self.rounds_indices * self.indices_block)
                self.indices_updates_size.set_as((self.indices_updates_size + self.updates_each_block - 1) 
                                                  // self.updates_each_block * self.updates_each_block)
                with self.tik_instance.if_scope(self.indices_updates_size <= self.indices_block):
                    self.compute_core_last_axis_unaligned(task, self.rounds_tail) # tail_block
                with self.tik_instance.else_scope():
                    self.compute_core_last_axis_unaligned_big(task, self.rounds_tail)
                    
            with self.tik_instance.else_scope():
                self.indices_updates_size.set_as(self.repeat_per_core * indices_width
                                                 + self.rounds_indices * self.indices_block)
                self.indices_updates_size.set_as((self.indices_updates_size + self.updates_each_block - 1)
                                                  // self.updates_each_block * self.updates_each_block)
                with self.tik_instance.if_scope(self.indices_updates_size <= self.indices_block):
                    self.compute_core_last_axis_unaligned(task, self.repeat_per_core) # tail_block
                with self.tik_instance.else_scope():
                    self.compute_core_last_axis_unaligned_big(task, self.repeat_per_core)

    def compute_core_last_axis_diff_shape_unaligned(self, task_id):
        """
            compute_core
            for last axis cases with unaligned data
            diffshape version
            e.g.:
            for data_width = 18
            -----------
            | 8 | 8 |2|
            -----------
            data_ub:
            ---------
            | 8 | 8 |
            ---------
            data_tail_ub:
                  -----
              8 2 | 8 |
                  -----
        """
        data_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_width",
                                              init_value=self.shape_data_ub[self.axis])
        block = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="block",
                                         init_value=self.data_block / self.repeat)
        tail_offset = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="tail_offset",
                                               init_value=(data_width - block))
        indices_width = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_width",
                                                 init_value=self.shape_indices_ub[self.axis])
        data_offset = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="data_offset")

        indices_col = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="indices_col")
        is_within_indices = self.tik_instance.Scalar(dtype=self.tiling_param_dtype, name="is_within_indices",
                                                     init_value=1)
        self.cal_indices_col_offset(task_id, indices_col, is_within_indices)

        data_ub = self.tik_instance.Tensor(self.dtype_data, [self.data_block], name="data_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub, self.data_gm[task_id * data_width], 0, 1, self.repeat - 1, 0, 0)
        data_tail_ub = self.tik_instance.Tensor(self.dtype_data, [block], name="data_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_tail_ub, self.data_gm[task_id * data_width + tail_offset], 0, 1, 1, 0, 0)

        with self.tik_instance.if_scope(is_within_indices == 1):
            self.indices_block_offset.set_as(indices_col * indices_width)
            with self.tik_instance.for_range(0, self.rounds_indices) as i:
                self.tik_instance.data_move(self.indices_ub,
                                            self.indices_gm[self.indices_block_offset + i * self.indices_block],
                                            0, 1, self.indices_repeat, 0, 0)
                self.tik_instance.data_move(self.updates_ub,
                                            self.updates_gm[self.indices_block_offset + i * self.indices_block],
                                            0, 1, self.updates_repeat, 0, 0)
                with self.tik_instance.for_range(0, self.indices_block) as offset:
                    data_offset.set_as(self.indices_ub[offset])
                    with self.tik_instance.if_scope(data_offset < tail_offset):
                        self.update_data(data_ub, self.updates_ub, data_offset, offset)
                    with self.tik_instance.else_scope():
                        self.update_data(data_tail_ub, self.updates_ub, data_offset - tail_offset, offset)
            with self.tik_instance.if_scope(self.tail_indices != 0):
                self.indices_block_offset.set_as(indices_col * indices_width + self.rounds_indices * self.indices_block)
                self.tik_instance.data_move(self.indices_ub, self.indices_gm[self.indices_block_offset],
                                            0, 1, self.tail_indices_repeat, 0, 0)
                self.tik_instance.data_move(self.updates_ub, self.updates_gm[self.indices_block_offset],
                                            0, 1, self.tail_updates_repeat, 0, 0)
                with self.tik_instance.for_range(0, self.tail_indices) as offset:
                    data_offset.set_as(self.indices_ub[offset])
                    with self.tik_instance.if_scope(data_offset < tail_offset):
                        self.update_data(data_ub, self.updates_ub, data_offset, offset)
                    with self.tik_instance.else_scope():
                        self.update_data(data_tail_ub, self.updates_ub, data_offset - tail_offset, offset)

        self.tik_instance.data_move(self.result_gm[task_id * data_width], data_ub, 0, 1, self.repeat - 1, 0, 0)
        self.tik_instance.data_move(self.result_gm[task_id * data_width + tail_offset], data_tail_ub, 0, 1, 1, 0, 0)

    def compute_core_last_axis_diff_shape_pre(self, task):
        with self.tik_instance.if_scope(self.shape_data_ub[self.axis] == self.data_block):
            self.compute_core_last_axis(task)
        with self.tik_instance.else_scope(): # unaligned cases
            self.compute_core_last_axis_diff_shape_unaligned(task)

    def scatter_elements_compute(self):
        self.get_tiling_args()
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_LASTAXIS_SAME_SHAPE):
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.compute_core_last_axis_pre(i + k * self.used_aicore_num)
                with self.tik_instance.if_scope(i < self.batch_tail):
                    self.compute_core_last_axis_pre(i + self.batch_num_per_aicore * self.used_aicore_num)
            with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_LASTAXIS_DIFF_SHAPE):
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.compute_core_last_axis_diff_shape_pre(i + k * self.used_aicore_num)
                with self.tik_instance.if_scope(i < self.batch_tail):
                    self.compute_core_last_axis_diff_shape_pre(i + self.batch_num_per_aicore * self.used_aicore_num)
            with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_SAMESHAPE):
                self.processor = ProcessorAxis(self.tik_instance, self.dtype_indices, self.params_row_indices,
                                               self.params_axis_indices, self.params_row_data, self.params_axis_data)
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.compute_core(i + k * self.used_aicore_num)
                with self.tik_instance.if_scope(i < self.batch_tail):
                    self.compute_core(i + self.batch_num_per_aicore * self.used_aicore_num)
            with self.tik_instance.else_scope(): # TILING_MODE_DIFFSHAPE
                self.processor = ProcessorNormal(self.tik_instance, self.shape_indices_ub, self.shape_data_ub,
                                                 self.dims_data, self.dtype_indices, self.dims_indices,
                                                 self.shape_acc_indices, self.axis)
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.compute_core(i + k * self.used_aicore_num)
                with self.tik_instance.if_scope(i < self.batch_tail):
                    self.compute_core(i + self.batch_num_per_aicore * self.used_aicore_num)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.data_gm, self.indices_gm, self.updates_gm],
                                   outputs=[self.result_gm],
                                   flowtable=[self.tiling_gm])
        tbe_context.get_context().add_compile_info('vars', {
            'ub_size_bytes': self.ub_size_bytes
        })
        return self.tik_instance


# 'pylint: disable=invalid-name
@register_operator('ScatterElements')
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def scatter_elements(data, indices, updates, result, axis = 0, reduction = "none", kernel_name="scatter_elements"):
    """
    Function: scatter_elements.
    Modify : 2023-04-24

    Init base parameters
    Parameters
    ----------
    input(data): dict
        data of input
    input(indices): dict
        data of input
    input(updates): dict
        data of input
    output(result): dict
        data of output
    attr(axis): int
        data of attr
    attr(reduction): str
        data of attr
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = ScatterElements(data, indices, updates, result, axis, reduction, kernel_name)
    return op_obj.scatter_elements_compute()
