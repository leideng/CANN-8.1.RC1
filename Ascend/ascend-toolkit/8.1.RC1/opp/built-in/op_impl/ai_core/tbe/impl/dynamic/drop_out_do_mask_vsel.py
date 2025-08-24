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
drop_out_do_mask_vsel.py
"""
from tbe.common.platform import get_bit_len
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=invalid-name
class Constant(object):
    """The class for constant"""
    # max int64
    MAX_INT64 = 2**63 - 1
    # ting param num
    TILING_ARG_NUM = 12
    # reserved ub size
    RESERVER_UB_SIZE = 4096
    # min num
    CORE_MINEST_NUM = 256
    # ub tiling: mask = x / 8
    EIGHT_NUM = 8
    # repeat limit
    REPEAT_LIMIT = 252
    # block size of uint8
    UINT8_BLOCK_SIZE = 32


class DropOutDoMaskVsel(object):
    """use to store dropoutdomask base parameters"""

    def __init__(self, x, mask, keep_prob, y, kernel_name):
        self.tik_instance = tik.Tik()

        # get input
        self.x_dtype = x.get("dtype").lower()
        self.mask_dtype = mask.get("dtype").lower()
        self.keep_prob_dtype = keep_prob.get("dtype").lower()
        self.y_dtype = y.get("dtype").lower()
        self.kernel_name = kernel_name

        para_check.check_dtype(self.mask_dtype, ("uint8", "uint1"), param_name="mask")
        para_check.check_dtype(self.x_dtype, ("float32", "float16", "bfloat16"), param_name="x")
        if self.keep_prob_dtype != self.x_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("DropOutDoMask", "keep_prob", "x",
                                                                  self.keep_prob_dtype, self.x_dtype)

        # get core info
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.x_size = get_bit_len(self.x_dtype) // 8
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.ub_ele = (self.ub_size - Constant.RESERVER_UB_SIZE) // self.x_size
        self.mask_len = 128 if self.x_dtype in ("float16",) else 64
        self.align_num = 2 if self.x_dtype in ("float16",) else 4
        self.repeat_limit = Constant.REPEAT_LIMIT // self.align_num * self.align_num

        # init gm
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="ting_gm",
                                                  scope=tik.scope_gm)
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, (Constant.MAX_INT64,), name="x_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_instance.Tensor("uint8", (Constant.MAX_INT64,), name="mask_gm", scope=tik.scope_gm)
        self.keep_prob_gm = self.tik_instance.Tensor(self.keep_prob_dtype, (Constant.MAX_INT64,),
                                                     name="keep_prob_gm",
                                                     scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.x_dtype, (Constant.MAX_INT64,), name="y_gm", scope=tik.scope_gm)

        # init scalar
        self.used_core_num = None
        self.num_per_core = None
        self.num_tail_core = None
        self.real_core_num = None
        self.ub_one_loop_num = None
        self.per_core_loop = None
        self.per_core_tail = None
        self.last_core_loop = None
        self.last_core_tail = None
        self.zero_scalar = None
        self.prob_rec = None

        # init ub
        self.mask_ub = None
        self.x_ub_in = None
        self.x_ub = None
        self.tiling_ub = None

    def _init_tiling_args(self):
        """get runtime tiling parameters from tiling"""
        self.used_core_num = self.tik_instance.Scalar("int64", name="core_used_num")
        self.num_per_core = self.tik_instance.Scalar("int64", name="num_per_core")
        self.num_tail_core = self.tik_instance.Scalar("int64", name="num_tail_core")
        self.real_core_num = self.tik_instance.Scalar("int64", name="core_num_scalar", init_value=self.core_num)
        self.ub_one_loop_num = self.tik_instance.Scalar("int64", name="ub_one_loop_num")
        self.per_core_loop = self.tik_instance.Scalar("int64", name="per_core_loop")
        self.per_core_tail = self.tik_instance.Scalar("int64", name="per_core_tail")
        self.last_core_loop = self.tik_instance.Scalar("int64", name="last_core_loop")
        self.last_core_tail = self.tik_instance.Scalar("int64", name="last_core_tail")
        self.used_core_num.set_as(self.tiling_ub[0])
        self.num_per_core.set_as(self.tiling_ub[1])
        self.num_tail_core.set_as(self.tiling_ub[2])
        self.real_core_num.set_as(self.tiling_ub[3])
        self.ub_one_loop_num.set_as(self.tiling_ub[4])
        self.per_core_loop.set_as(self.tiling_ub[5])
        self.per_core_tail.set_as(self.tiling_ub[6])
        self.last_core_loop.set_as(self.tiling_ub[7])
        self.last_core_tail.set_as(self.tiling_ub[8])

    def _init_ub_tensor(self):
        """compute the ub size of tensors"""
        self.mask_ub = self.tik_instance.Tensor(
            "uint8", ((self.ub_one_loop_num + Constant.EIGHT_NUM - 1) // Constant.EIGHT_NUM,),
            name="mask_ub",
            scope=tik.scope_ubuf)

        if self.x_dtype == "bfloat16":
            self.x_ub_in = self.tik_instance.Tensor("bfloat16", (self.ub_one_loop_num,),
                                                    name="x_ub_in",
                                                    scope=tik.scope_ubuf)
            self.x_ub = self.tik_instance.Tensor("float32", (self.ub_one_loop_num,), name="x_ub", scope=tik.scope_ubuf)
            self.zero_scalar = self.tik_instance.Scalar("float32", name="zeros_scalar", init_value=0.0)
        else:
            self.x_ub = self.tik_instance.Tensor(self.x_dtype, (self.ub_one_loop_num,),
                                                 name="x_ub",
                                                 scope=tik.scope_ubuf)
            self.zero_scalar = self.tik_instance.Scalar(self.x_dtype, name="zeros_scalar", init_value=0.0)

    def _init_prob_scalar(self):
        """ _init_prob_scalar"""
        prob_dtype = "float32" if self.x_dtype in ("bfloat16",) else self.keep_prob_dtype
        self.prob_rec = self.tik_instance.Scalar(prob_dtype, name="keep_prob_scaler")
        vcetor_num = 128 if self.x_dtype in ("float16", "bfloat16") else 64
        with self.tik_instance.new_stmt_scope():
            if self.x_dtype == "bfloat16":
                keep_prob_ub_in = self.tik_instance.Tensor("bfloat16", (vcetor_num,),
                                                           name="keep_prob_ub_in",
                                                           scope=tik.scope_ubuf)
                keep_prob_ub = self.tik_instance.Tensor("float32", (vcetor_num,),
                                                        name="keep_prob_ub",
                                                        scope=tik.scope_ubuf)
                one_vdiv_ub = self.tik_instance.Tensor("float32", (vcetor_num,),
                                                       name="one_vdiv_ub",
                                                       scope=tik.scope_ubuf)
                self.tik_instance.data_move(keep_prob_ub_in, self.keep_prob_gm, 0, 1, 1, 0, 0)
                self.tik_instance.vconv(1, "", keep_prob_ub, keep_prob_ub_in, 1, 1, 1, 8, 4)
            else:
                keep_prob_ub = self.tik_instance.Tensor(self.keep_prob_dtype, (vcetor_num,),
                                                        name="keep_prob_ub",
                                                        scope=tik.scope_ubuf)
                self.tik_instance.data_move(keep_prob_ub, self.keep_prob_gm, 0, 1, 1, 0, 0)
                one_vdiv_ub = self.tik_instance.Tensor(self.keep_prob_dtype, (vcetor_num,),
                                                       name="one_vdiv_ub",
                                                       scope=tik.scope_ubuf)

            self.tik_instance.vector_dup(1, one_vdiv_ub, 1.0, 1, 1, 8)
            self.tik_instance.vdiv(1, keep_prob_ub, one_vdiv_ub, keep_prob_ub, 1, 1, 1, 1, 8, 8, 8)
            self.prob_rec.set_as(keep_prob_ub[0])

    def _calc(self, size):
        """calc function for float16 float32"""
        with self.tik_instance.if_scope(size > 0):
            size_loop = (size + self.mask_len - 1) // self.mask_len
            repeat_loop = size_loop // self.repeat_limit
            repeat_left = size_loop % self.repeat_limit
            repeat_left_align = repeat_left // self.align_num * self.align_num
            repeat_left_tail = repeat_left % self.align_num

            with self.tik_instance.for_range(0, repeat_loop) as loop_idx:
                repeat_offset = loop_idx * self.repeat_limit * self.mask_len
                self.tik_instance.vmuls(self.mask_len, self.x_ub[repeat_offset], self.x_ub[repeat_offset],
                                        self.prob_rec, self.repeat_limit, 1, 1, 8, 8)
                self.tik_instance.vsel(self.mask_len, 1, self.x_ub[repeat_offset],
                                       self.mask_ub[repeat_offset // Constant.EIGHT_NUM], self.x_ub[repeat_offset],
                                       self.zero_scalar, self.repeat_limit, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.if_scope(repeat_left_align > 0):
                repeat_offset = repeat_loop * self.repeat_limit * self.mask_len
                self.tik_instance.vmuls(self.mask_len, self.x_ub[repeat_offset], self.x_ub[repeat_offset],
                                        self.prob_rec, repeat_left_align, 1, 1, 8, 8)
                self.tik_instance.vsel(self.mask_len, 1, self.x_ub[repeat_offset],
                                       self.mask_ub[repeat_offset // Constant.EIGHT_NUM], self.x_ub[repeat_offset],
                                       self.zero_scalar, repeat_left_align, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.if_scope(repeat_left_tail > 0):
                repeat_offset = repeat_loop * self.repeat_limit * self.mask_len + repeat_left_align * self.mask_len
                self.tik_instance.vmuls(self.mask_len, self.x_ub[repeat_offset], self.x_ub[repeat_offset],
                                        self.prob_rec, repeat_left_tail, 1, 1, 8, 8)
                self.tik_instance.vsel(self.mask_len, 1, self.x_ub[repeat_offset],
                                       self.mask_ub[repeat_offset // Constant.EIGHT_NUM], self.x_ub[repeat_offset],
                                       self.zero_scalar, repeat_left_tail, 1, 1, 1, 8, 8, 8)

    def _run_one_loop(self, gm_offset, process_num):
        """_run_one_loop"""
        if self.x_dtype == "bfloat16":
            self.tik_instance.data_move_pad(self.x_ub_in, self.x_gm[gm_offset], 1, process_num * self.x_size, 0, 0)
            self.tik_instance.data_move_pad(self.mask_ub, self.mask_gm[gm_offset // 8], 1,
                                            ((process_num + Constant.EIGHT_NUM - 1) // Constant.EIGHT_NUM), 0, 0)
            self._calc_bf16(process_num)
            self.tik_instance.data_move_pad(self.y_gm[gm_offset], self.x_ub_in, 1, process_num * self.x_size, 0, 0)
        else:
            self.tik_instance.data_move_pad(self.x_ub, self.x_gm[gm_offset], 1, process_num * self.x_size, 0, 0)
            self.tik_instance.data_move_pad(self.mask_ub, self.mask_gm[gm_offset // 8], 1,
                                            ((process_num + Constant.EIGHT_NUM - 1) // Constant.EIGHT_NUM), 0, 0)
            self._calc(process_num)
            self.tik_instance.data_move_pad(self.y_gm[gm_offset], self.x_ub, 1, process_num * self.x_size, 0, 0)

    def _run_one_core(self, _core_idx, loop, tail):
        """_run_one_core"""
        # process algin ub_one_loop_num
        with self.tik_instance.for_range(0, loop) as loop_idx:
            copy_gm_offset = _core_idx * self.num_per_core + loop_idx * self.ub_one_loop_num
            self._run_one_loop(copy_gm_offset, self.ub_one_loop_num)

        with self.tik_instance.if_scope(tail > 0):
            copy_gm_offset = _core_idx * self.num_per_core + loop * self.ub_one_loop_num
            self._run_one_loop(copy_gm_offset, tail)

    def _drop_do_mask_main(self):
        """_drop_do_mask_main"""
        # get tiling
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
        self._init_tiling_args()

        # do tiling
        with self.tik_instance.for_range(0, self.real_core_num, block_num=self.real_core_num) as _core_idx:
            self._init_ub_tensor()
            self._init_prob_scalar()

            with self.tik_instance.if_scope(_core_idx < self.used_core_num - 1):
                self._run_one_core(_core_idx, self.per_core_loop, self.per_core_tail)
            with self.tik_instance.if_scope(_core_idx == self.used_core_num - 1):
                self._run_one_core(_core_idx, self.last_core_loop, self.last_core_tail)

    def drop_do_mask_operator(self):
        """drop_do_mask_operator"""
        self._drop_do_mask_main()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {"ub_ele": self.ub_ele})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x_gm, self.mask_gm, self.keep_prob_gm),
                                   outputs=(self.y_gm,),
                                   flowtable=(self.tiling_gm,),
                                   config=opt_config)

    def _calc_bf16(self, size):
        """calc function for bfloat16"""
        round_mode = "round"
        with self.tik_instance.if_scope(size > 0):
            size_loop = (size + self.mask_len - 1) // self.mask_len
            repeat_loop = size_loop // self.repeat_limit
            repeat_left = size_loop % self.repeat_limit
            repeat_left_align = repeat_left // self.align_num * self.align_num
            repeat_left_tail = repeat_left % self.align_num
            with self.tik_instance.for_range(0, repeat_loop) as loop_idx:
                repeat_offset = loop_idx * self.repeat_limit * self.mask_len
                self.tik_instance.vconv(self.mask_len, "", self.x_ub[repeat_offset], self.x_ub_in[repeat_offset],
                                        self.repeat_limit, 1, 1, 8, 4)
                self.tik_instance.vmuls(self.mask_len, self.x_ub[repeat_offset], self.x_ub[repeat_offset],
                                        self.prob_rec, self.repeat_limit, 1, 1, 8, 8)
                self.tik_instance.vsel(self.mask_len, 1, self.x_ub[repeat_offset], 
                                       self.mask_ub[repeat_offset // Constant.EIGHT_NUM], self.x_ub[repeat_offset],
                                       self.zero_scalar, self.repeat_limit, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vconv(self.mask_len, round_mode, self.x_ub_in[repeat_offset],
                                        self.x_ub[repeat_offset], self.repeat_limit, 1, 1, 4, 8)
            with self.tik_instance.if_scope(repeat_left_align > 0):
                repeat_offset = repeat_loop * self.repeat_limit * self.mask_len
                self.tik_instance.vconv(self.mask_len, "", self.x_ub[repeat_offset], self.x_ub_in[repeat_offset],
                                        repeat_left_align, 1, 1, 8, 4)
                self.tik_instance.vmuls(self.mask_len, self.x_ub[repeat_offset], self.x_ub[repeat_offset],
                                        self.prob_rec, repeat_left_align, 1, 1, 8, 8)
                self.tik_instance.vsel(self.mask_len, 1, self.x_ub[repeat_offset],
                                       self.mask_ub[repeat_offset // Constant.EIGHT_NUM], self.x_ub[repeat_offset],
                                       self.zero_scalar, repeat_left_align, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vconv(self.mask_len, round_mode, self.x_ub_in[repeat_offset],
                                        self.x_ub[repeat_offset], repeat_left_align, 1, 1, 4, 8)
            with self.tik_instance.if_scope(repeat_left_tail > 0):
                repeat_offset = repeat_loop * self.repeat_limit * self.mask_len + repeat_left_align * self.mask_len
                self.tik_instance.vconv(self.mask_len, "", self.x_ub[repeat_offset], self.x_ub_in[repeat_offset],
                                        repeat_left_tail, 1, 1, 8, 4)
                self.tik_instance.vmuls(self.mask_len, self.x_ub[repeat_offset], self.x_ub[repeat_offset],
                                        self.prob_rec, repeat_left_tail, 1, 1, 8, 8)
                self.tik_instance.vsel(self.mask_len, 1, self.x_ub[repeat_offset],
                                       self.mask_ub[repeat_offset // Constant.EIGHT_NUM], self.x_ub[repeat_offset],
                                       self.zero_scalar, repeat_left_tail, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vconv(self.mask_len, round_mode, self.x_ub_in[repeat_offset],
                                        self.x_ub[repeat_offset], repeat_left_tail, 1, 1, 4, 8)


def drop_out_do_mask_vsel(x, mask, keep_prob, y, kernel_name="drop_out_do_mask_vsel"):
    """
    drop_out_do_mask_vsel
    """
    obj = DropOutDoMaskVsel(x, mask, keep_prob, y, kernel_name)
    obj.drop_do_mask_operator()
