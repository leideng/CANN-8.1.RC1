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
dynamic diag
"""
from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    #tiling arg num
    TILING_ARG_NUM = 8
    #MAX INT 64
    MAX_INT64 = 2 ** 64 - 1
    #NUM 64
    NUM_64 = 64
    #NUM 128
    NUM_128 = 128
    #MAX BURST LEN
    MAX_BURST_LEN = 65535
    #MAX uint32
    MAX_UINT32 = 4294967295
    # UINT16
    UINT16 = "uint16"
    # UINT32
    UINT32 = "uint32"


# 'pylint: disable=unused-argument,invalid-name,useless-object-inheritance,too-many-lines,too-many-arguments
# 'pylint: disable=too-many-statements,too-many-locals,attribute-defined-outside-init,too-many-instance-attributes
# 'pylint: disable=missing-class-docstring,missing-function-docstring,too-many-function-args
class Diag():
    """
    Function: class that execute Diag
    """
    def __init__(self, input_x, kernel_name="diag"):
        self.shape_x = input_x.get("shape")
        self.dtype_x = input_x.get("dtype")
        self.assist_dtype = Constant.UINT16 if self.dtype_x in ("float16") else Constant.UINT32

        # check dtype
        para_check.check_dtype(self.dtype_x, ("float16", "float32", "int32"), param_name="value")

        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.compute_assist()

        self.tiling_input_num = self.tik_instance.Scalar("int64", name="tiling_input_num", init_value=0)
        self.tiling_act_core_num = self.tik_instance.Scalar("int64", name="tiling_act_core_num", init_value=1)
        self.tiling_each_core_num = self.tik_instance.Scalar("int64", name="tiling_each_core_num", init_value=1)
        self.tiling_last_core_num = self.tik_instance.Scalar("int64", name="tiling_last_core_num", init_value=1)

        #get ai_core num
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.core_num_scalar = self.tik_instance.Scalar("int64", name="core_num_scalar", init_value=self.ai_core_num)

        #assist space
        self.dtype_bytes_size = common_util.get_data_size(self.dtype_x)
        self.dtype_assist_bytes_size = self.dtype_bytes_size * Constant.NUM_64 * Constant.NUM_64
        self.dtype_vmul_out_bytes_size = self.dtype_assist_bytes_size
        dtype_tiling_bytes_size = common_util.get_data_size("int64") * Constant.TILING_ARG_NUM

        #get ub size
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - \
                             self.dtype_assist_bytes_size - self.dtype_vmul_out_bytes_size - \
                             dtype_tiling_bytes_size

        #get data_num in one block
        self.data_each_block = constant.BLOCK_SIZE // self.dtype_bytes_size

        #ub for input and output
        self.ub_tensor_size = (self.ub_size_bytes // self.dtype_bytes_size // 2 // self.data_each_block *
                               self.data_each_block)

        #make gm tensor
        self.input_x_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64,), name="input_x_gm", \
        scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64,),
                                                  name="output_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_gm", \
        scope=tik.scope_gm)
    
        self.input_x_ub = None
        self.vmul_out = None
        self.assist_ub = None

        self.data_move_pad_support = tbe_platform.api_check_support("tik.data_move_pad")

    @staticmethod
    def ceil(data_a, data_b):
        """
        func ceil
        """
        res = (data_a + data_b - 1) // data_b
        return res

    def diag_compute_tiling(self):
        """
        tiling info:
            tiling_input_num: number of input_x
            tiling_act_core_num: need use aicore number
            tiling_each_core_num: number of one each aicore need to compute except last aicore
            tiling_last_core_num: number of one last aicore need to compute
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_ub", \
        scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self.tiling_input_num.set_as(self.tiling_ub[0])
        self.tiling_act_core_num.set_as(self.tiling_ub[1])
        self.tiling_each_core_num.set_as(self.tiling_ub[2])
        self.tiling_last_core_num.set_as(self.tiling_ub[3])
        self.core_num_scalar.set_as(self.tiling_ub[4])

    def compute_assist(self):
        """
        func compute_assist
        """
        assist_data = [[0] * 64 for _ in range(64)]
        for i in range(0, 64):
            assist_data[i][i] = Constant.MAX_BURST_LEN if self.assist_dtype == Constant.UINT16 else Constant.MAX_UINT32
        self.assist_gm = self.tik_instance.Tensor(self.assist_dtype, 
                                                  (Constant.NUM_64, Constant.NUM_64), name="assist_gm",
                                                  scope=tik.scope_gm, init_value=assist_data)

    def core_scedule_args(self, core_idx):
        """
        core_scedule_args
        """
        with self.tik_instance.if_scope(core_idx < self.tiling_act_core_num - 1):
            move_offset = core_idx * self.tiling_each_core_num
            loop_num = self.tiling_each_core_num // Constant.NUM_64
            burst_len = Constant.NUM_64 // self.data_each_block
            self.tik_instance.data_move(self.assist_ub,
                                        self.assist_gm,
                                        0, Constant.NUM_64, burst_len, 0, 0)
            nburst_x = self.ceil(self.tiling_each_core_num // self.data_each_block, Constant.MAX_BURST_LEN)
            burst_len_x = self.ceil(self.tiling_each_core_num // self.data_each_block, nburst_x)
            with self.tik_instance.new_stmt_scope():
                if self.data_move_pad_support:
                    self.tik_instance.data_move_pad(self.input_x_ub, self.input_x_gm[move_offset],
                                                    nburst_x, self.tiling_each_core_num * self.dtype_bytes_size, 0, 0)
                else:
                    self.tik_instance.data_move(self.input_x_ub, self.input_x_gm[move_offset],
                                                0, nburst_x, burst_len_x, 0, 0)
                with self.tik_instance.for_range(0, loop_num) as loop_index:
                    move_offset = loop_index * Constant.NUM_64
                    compute_num = Constant.NUM_64
                    move_offset_output = core_idx * self.tiling_each_core_num * self.tiling_input_num + \
                                        loop_index * self.tiling_input_num * Constant.NUM_64 + \
                                        core_idx * self.tiling_each_core_num + \
                                        loop_index * Constant.NUM_64
                    self.diag_compute_each_loop(move_offset, compute_num, move_offset_output)

        with self.tik_instance.if_scope(core_idx == self.tiling_act_core_num - 1):
            move_offset = core_idx * self.tiling_each_core_num
            burst_len = Constant.NUM_64 // self.data_each_block
            self.tik_instance.data_move(self.assist_ub, self.assist_gm,
                                        0, Constant.NUM_64, burst_len, 0, 0)
            #last_num enough 64
            loop_num = self.tiling_last_core_num // 64
            with self.tik_instance.if_scope(loop_num > 0):
                move_offset = core_idx * self.tiling_each_core_num
                nburst_x = self.ceil(self.tiling_last_core_num, self.data_each_block)
                nburst_x = self.ceil(nburst_x, Constant.MAX_BURST_LEN)
                burst_len_x = self.ceil(self.tiling_last_core_num // self.data_each_block, nburst_x)
                with self.tik_instance.new_stmt_scope():
                    if self.data_move_pad_support:
                        self.tik_instance.data_move_pad(self.input_x_ub,
                                                        self.input_x_gm[move_offset],
                                                        nburst_x, self.tiling_last_core_num * self.dtype_bytes_size, 0, 0)
                    else:
                        self.tik_instance.data_move(self.input_x_ub,
                                                    self.input_x_gm[move_offset],
                                                    0, nburst_x, burst_len_x, 0, 0)
                    with self.tik_instance.for_range(0, loop_num) as last_core_index:
                        move_offset = last_core_index * Constant.NUM_64
                        compute_num = Constant.NUM_64
                        move_offset_output = core_idx * self.tiling_each_core_num * self.tiling_input_num + \
                                            last_core_index * self.tiling_input_num * Constant.NUM_64 + \
                                            core_idx * self.tiling_each_core_num + last_core_index * Constant.NUM_64
                        self.diag_compute_each_loop(move_offset, compute_num, move_offset_output)
            #last_num not enough 64
            with self.tik_instance.if_scope((self.tiling_last_core_num % Constant.NUM_64) != 0):
                with self.tik_instance.if_scope(self.tiling_input_num >= Constant.NUM_64):
                    move_offset = core_idx * self.tiling_each_core_num
                    back_offset = Constant.NUM_64 - (self.tiling_last_core_num % Constant.NUM_64)
                    nburst_x = self.ceil(self.tiling_last_core_num, self.data_each_block)
                    nburst_x = self.ceil(nburst_x, Constant.MAX_BURST_LEN)
                    burst_len_x = self.ceil(self.tiling_last_core_num + back_offset, self.data_each_block)
                    burst_len_x = self.ceil(burst_len_x, nburst_x)
                    with self.tik_instance.new_stmt_scope():
                        if self.data_move_pad_support:
                            self.tik_instance.data_move_pad(self.input_x_ub,
                                                            self.input_x_gm[move_offset - back_offset],
                                                            nburst_x, (self.tiling_last_core_num + back_offset) * self.dtype_bytes_size, 
                                                            0, 0)
                        else:
                            self.tik_instance.data_move(self.input_x_ub,
                                                        self.input_x_gm[move_offset - back_offset],
                                                        0, nburst_x, burst_len_x, 0, 0)
                        last_num = self.tiling_last_core_num % Constant.NUM_64
                        back_offset = Constant.NUM_64 - last_num
                        move_offset = loop_num * Constant.NUM_64
                        compute_num = Constant.NUM_64
                        move_offset_output = core_idx * self.tiling_each_core_num * self.tiling_input_num + \
                                            loop_num * self.tiling_input_num * Constant.NUM_64 - \
                                            back_offset * self.tiling_input_num + \
                                            core_idx * self.tiling_each_core_num + \
                                            (loop_num - 1) * Constant.NUM_64 + last_num
                        self.diag_compute_each_loop(move_offset, compute_num, move_offset_output)

                with self.tik_instance.if_scope(self.tiling_input_num < Constant.NUM_64):
                    burst_len_x = self.ceil(self.tiling_input_num, self.data_each_block)
                    with self.tik_instance.new_stmt_scope():
                        if self.data_move_pad_support:
                            self.tik_instance.data_move_pad(self.input_x_ub,
                                                            self.input_x_gm,
                                                            1, self.tiling_input_num * self.dtype_bytes_size, 0, 0)
                        else:
                            self.tik_instance.data_move(self.input_x_ub,
                                                        self.input_x_gm,
                                                        0, 1, burst_len_x, 0, 0)
                        data_x_ub = self.tik_instance.Tensor(self.dtype_x,
                                                             (self.tiling_input_num, self.tiling_input_num),
                                                             name="data_x_ub", scope=tik.scope_ubuf)
                        burst_len = self.ceil(self.tiling_input_num * self.tiling_input_num, self.data_each_block)
                        if self.data_move_pad_support:
                            self.tik_instance.data_move_pad(data_x_ub,
                                                            self.output_gm,
                                                            1, self.tiling_input_num * self.tiling_input_num * self.dtype_bytes_size, 
                                                            0, 0)
                        else:
                            self.tik_instance.data_move(data_x_ub,
                                                        self.output_gm,
                                                        0, 1, burst_len, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_input_num) as loop_index:
                            data_index = loop_index * (self.tiling_input_num + 1)
                            data = self.tik_instance.Scalar(self.dtype_x, name="input_x_data",
                                                            init_value=self.input_x_ub[loop_index])
                            data_x_ub[data_index].set_as(data)
                        if self.data_move_pad_support:
                            self.tik_instance.data_move_pad(self.output_gm,
                                                            data_x_ub,
                                                            1, self.tiling_input_num * self.tiling_input_num * self.dtype_bytes_size, 
                                                            0, 0)
                        else:
                            self.tik_instance.data_move(self.output_gm,
                                                        data_x_ub,
                                                        0, 1, burst_len, 0, 0)
                            
    def diag_compute(self):
        """
        func diag_compute
        """
        with self.tik_instance.new_stmt_scope():
            self.diag_compute_tiling()
        with self.tik_instance.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar) as index:
            # make ub tensor
            self.input_x_ub = self.tik_instance.Tensor(self.dtype_x, (self.ub_tensor_size,),
                                                       name="input_x_ub", scope=tik.scope_ubuf)
            self.vmul_out = self.tik_instance.Tensor(self.dtype_x, (Constant.NUM_64, Constant.NUM_64),
                                                     name="vmul_out", scope=tik.scope_ubuf)
            self.assist_ub = self.tik_instance.Tensor(self.assist_dtype, (Constant.NUM_64, Constant.NUM_64),
                                                      name="assist_ub", scope=tik.scope_ubuf)

            burst_assist = Constant.NUM_64 * Constant.NUM_64 // self.data_each_block
            self.tik_instance.data_move(self.assist_ub, self.assist_gm, 0, 1, burst_assist, 0, 0)

            self.core_scedule_args(index)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        tbe_context.get_context().add_compile_info(
            "vars", {"core_num": self.ai_core_num}
        )
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_x_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)

        return self.tik_instance

    def diag_compute_each_loop(self, move_offset, move_num, move_offset_output):
        """
        func compute_each_loop
        """
        burst_len = move_num // self.data_each_block
        rep_stride = move_num // self.data_each_block
        repeat_time = move_num * move_num // (rep_stride * self.data_each_block)
        vmul_out_uint16 = self.vmul_out.reinterpret_cast_to(Constant.UINT16)
        input_x_ub_uint16 = self.input_x_ub.reinterpret_cast_to(Constant.UINT16)
        assist_ub_unit16 = self.assist_ub if self.assist_dtype == Constant.UINT16 else \
            self.assist_ub.reinterpret_cast_to(Constant.UINT16)
        vand_mask = Constant.NUM_64 if self.assist_dtype == Constant.UINT16 else Constant.NUM_128
        move_offset = move_offset if self.assist_dtype == Constant.UINT16 else move_offset * 2
        self.tik_instance.vand(vand_mask,
                               vmul_out_uint16,
                               input_x_ub_uint16[move_offset],
                               assist_ub_unit16,
                               repeat_time, 1, 1, 1, rep_stride, 0, rep_stride)
        self.vmul_out = vmul_out_uint16.reinterpret_cast_to(self.dtype_x)

        with self.tik_instance.if_scope(self.tiling_input_num % self.data_each_block == 0):
            dst_stride = self.tiling_input_num // self.data_each_block - Constant.NUM_64 // self.data_each_block
            burst = Constant.NUM_64 // self.data_each_block
            self.tik_instance.data_move(self.output_gm[move_offset_output], self.vmul_out,
                                        0, Constant.NUM_64, burst, 0, dst_stride)

        with self.tik_instance.if_scope(self.tiling_input_num % self.data_each_block != 0):
            burst_len = Constant.NUM_64 // self.data_each_block
            with self.tik_instance.for_range(0, Constant.NUM_64) as offset_index:
                self.tik_instance.data_move(self.output_gm[move_offset_output + offset_index * self.tiling_input_num],
                                            self.vmul_out[offset_index * Constant.NUM_64], 0, 1, burst_len, 0, 0)


@register_operator("Diag")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def diag(x, y, kernel_name="diag"):
    """
    algorithm: diag
    calculating diag(x):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x, and the dtype of x must
        be in [float16, float32, int32]
    y: dict
        dict with keys(shape and dtype) of y
    kernel_name: str
        kernel name, default value is "diag"

    Returns
    -------
    None
    """
    #check shape
    if len(x.get("shape")) > 4:
        error_detail = "length of x'shape should be less than 5 but got: %d" % len(x.get("shape"))
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
    obj = Diag(x, kernel_name)
    tik_instance = obj.diag_compute()
    return tik_instance
