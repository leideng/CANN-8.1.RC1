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
yuv444_2_yuv422
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_common


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
class Yuv4442Yuv422():
    """
    Yuv4442Yuv422 class
    """
    MAX_INT32 = 2 ** 31 - 1
    MASK_F16 = 128
    MASK_U8 = 256
    BLOCK_I64 = 4
    BLOCK_F16 = 16
    BLOCK_U8 = 32

    IN_C = 4
    OUT_C = 2
    TILING_ARG_NUM = 8
    # reserved ub size
    RESERVED_UB_SIZE = 2 * 1024
    REPEATS_254 = 254
    # src1_pattern params for vreduce
    YUV422_PATTERN_SCALAR = 2 ** 0 + 2 ** 1 + 2 ** 4 + 2 ** 6 + 2 ** 8 + 2 ** 9 + 2 ** 12 + 2 ** 14

    def __init__(self, x, y):
        """
        init
        """
        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.RESERVED_UB_SIZE
        dsize = 2
        self.half_ub_elems = self.ub_size // dsize // 2
        if self.half_ub_elems >= self.REPEATS_254 * self.MASK_F16:
            self.one_loop_elems = self.REPEATS_254 * self.MASK_F16
        else:
            self.one_loop_elems = self.half_ub_elems // self.MASK_U8 * self.MASK_U8
        self.one_loop_pixels = self.one_loop_elems // self.IN_C

        self.x_dtype = x.get("dtype").lower()
        self.y_dtype = y.get("dtype").lower()
        self.dtype_i32 = "int32"
        self.tiling_dtype = "int64"
        self.tiling_align = util_common.align(self.TILING_ARG_NUM, self.BLOCK_I64)

        self.x_gm = None
        self.y_gm = None
        self.tiling_gm = None
        self._init_gm_tensor()
        self.yuv422_pattern = None

        # tiling params
        self.h = None
        self.w = None
        self.need_core_num = None
        self.pixel_num_low = None
        self.pixel_num_last = None

    def yuv444_2_yuv422_compute(self):
        """
        compute of yuv444 to yuv422
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            # get tiling data
            self._get_tiling_args()

            with self.tik_inst.if_scope(core_id < self.need_core_num):
                self._init_ub_tensor()

                with self.tik_inst.if_scope(core_id < self.need_core_num - 1):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.pixel_num_low * core_id
                        self._one_core_compute(start_idx, self.pixel_num_low)

                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.pixel_num_low * (self.need_core_num - 1)
                        self._one_core_compute(start_idx, self.pixel_num_last)

    def get_inputs_outputs_gm(self):
        inputs_gm = (self.x_gm,)
        outputs_gm = (self.y_gm,)

        return inputs_gm, outputs_gm

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        self.x_gm = self.tik_inst.Tensor(self.x_dtype, (self.MAX_INT32,), name="x_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_inst.Tensor(self.y_dtype, (self.MAX_INT32,), name="y_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)

    def _init_ub_tensor(self):
        """
        init ub tensor
        """
        self.yuv422_pattern = self.tik_inst.Tensor("uint16", (8,), name="yuv422_pattern", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(8, self.yuv422_pattern, self.YUV422_PATTERN_SCALAR, 1, 0, 0)

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.h = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h")
        self.w = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w")
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.pixel_num_low = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pixel_num_low")
        self.pixel_num_last = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pixel_num_last")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.BLOCK_I64, 0, 0)

            self.h.set_as(tiling_ub[0])
            self.w.set_as(tiling_ub[1])
            self.need_core_num.set_as(tiling_ub[2])
            self.pixel_num_low.set_as(tiling_ub[3])
            self.pixel_num_last.set_as(tiling_ub[4])

    def _tail_process(self, pixel_start_idx, loop, tail):
        """
        tail process
        """
        with self.tik_inst.new_stmt_scope():
            pixel_offset = pixel_start_idx + self.one_loop_pixels * loop
            x_offset = pixel_offset * self.IN_C
            y_offset = pixel_offset * self.OUT_C
            tail_in = tail * self.IN_C
            tail_out = tail * self.OUT_C

            yuv422_elems = self.one_loop_elems // 2
            yuv422_fp16_ub = self.tik_inst.Tensor(self.x_dtype, (yuv422_elems + self.MASK_U8,),
                                                  name="yuv422_fp16_ub", scope=tik.scope_ubuf)
            with self.tik_inst.new_stmt_scope():
                x_ub = self.tik_inst.Tensor(self.x_dtype, (self.one_loop_elems + self.MASK_F16,),
                                            name="x_ub", scope=tik.scope_ubuf)
                self.tik_inst.data_move(x_ub, self.x_gm[x_offset], 0, 1,
                                        util_common.ceil(tail_in, self.BLOCK_F16), 0, 0)
                self.tik_inst.vreduce(self.MASK_F16, yuv422_fp16_ub, x_ub, self.yuv422_pattern,
                                      util_common.ceil(tail_in, self.MASK_F16), 1, 8, 0, mask_mode="normal")

            yuv422_u8_ub = self.tik_inst.Tensor(self.y_dtype, (yuv422_elems + self.MASK_U8,),
                                                name="yuv422_u8_ub", scope=tik.scope_ubuf)
            self.tik_inst.vconv(self.MASK_F16, "", yuv422_u8_ub, yuv422_fp16_ub,
                                util_common.ceil(tail_out, self.MASK_F16), 1, 1, 4, 8)

            # move result to gm
            with self.tik_inst.if_scope(tik.any(tail_out % self.BLOCK_U8 == 0, tail_out < self.BLOCK_U8)):
                self.tik_inst.data_move(self.y_gm[y_offset], yuv422_u8_ub, 0, 1,
                                        util_common.ceil(tail_out, self.BLOCK_U8), 0, 0)
            with self.tik_inst.else_scope():
                block_ub = self.tik_inst.Tensor(self.y_dtype, (self.BLOCK_U8,), name="block_ub", scope=tik.scope_ubuf)
                last_block_offset = tail_out - self.BLOCK_U8
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    for i in range(self.BLOCK_U8):
                        block_ub[i].set_as(yuv422_u8_ub[last_block_offset + i])
                self.tik_inst.data_move(self.y_gm[y_offset], yuv422_u8_ub, 0, 1, tail_out // self.BLOCK_U8, 0, 0)
                self.tik_inst.data_move(self.y_gm[y_offset + last_block_offset], block_ub, 0, 1, 1, 0, 0)

    def _one_loop_process(self, pixel_start_idx, loop_idx):
        """
        one loop process
        """
        with self.tik_inst.new_stmt_scope():
            pixel_offset = pixel_start_idx + self.one_loop_pixels * loop_idx
            x_offset = pixel_offset * self.IN_C
            y_offset = pixel_offset * self.OUT_C
            yuv422_elems = self.one_loop_elems // 2
            yuv422_fp16_ub = self.tik_inst.Tensor(self.x_dtype, (yuv422_elems,), name="yuv422_fp16_ub",
                                                  scope=tik.scope_ubuf)
            with self.tik_inst.new_stmt_scope():
                x_ub = self.tik_inst.Tensor(self.x_dtype, (self.one_loop_elems,), name="x_ub", scope=tik.scope_ubuf)
                self.tik_inst.data_move(x_ub, self.x_gm[x_offset], 0, 1, self.one_loop_elems // self.BLOCK_F16, 0, 0)
                self.tik_inst.vreduce(self.MASK_F16, yuv422_fp16_ub, x_ub, self.yuv422_pattern,
                                      self.one_loop_elems // self.MASK_F16, 1, 8, 0, mask_mode="normal")

            yuv422_u8_ub = self.tik_inst.Tensor(self.y_dtype, (yuv422_elems,), name="yuv422_u8_ub",
                                                scope=tik.scope_ubuf)
            self.tik_inst.vconv(self.MASK_F16, "", yuv422_u8_ub, yuv422_fp16_ub, yuv422_elems // self.MASK_F16,
                                1, 1, 4, 8)

            self.tik_inst.data_move(self.y_gm[y_offset], yuv422_u8_ub, 0, 1, yuv422_elems // self.BLOCK_U8, 0, 0)

    def _one_core_compute(self, pixel_start_idx, pixel_num):
        """
        compute for one core
        """
        loop = self.tik_inst.Scalar(dtype=self.dtype_i32, name="loop")
        tail = self.tik_inst.Scalar(dtype=self.dtype_i32, name="tail")
        loop.set_as(pixel_num // self.one_loop_pixels)
        tail.set_as(pixel_num % self.one_loop_pixels)
        with self.tik_inst.if_scope(tik.all(tail > 0, tail < 16, loop > 0)):
            loop.set_as(loop - 1)
            tail.set_as(self.one_loop_pixels + tail)

        with self.tik_inst.for_range(0, loop) as loop_idx:
            self._one_loop_process(pixel_start_idx, loop_idx)
        with self.tik_inst.if_scope(tail > 0):
            self._tail_process(pixel_start_idx, loop, tail)


def _check_input_params(x, y):
    """
    check input parameters.
    """
    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    para_check.check_dtype(x_dtype, ("float16",), param_name="x")
    para_check.check_dtype(y_dtype, ("uint8",), param_name="y")


@register_operator("YUV4442YUV422")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def yuv444_2_yuv422(x, y, kernel_name="yuv444_2_yuv422"):
    """
    YUV4442YUV422 op

    Parameters
    ----------
    x: dict
        the dict of input yuv444, shape is [h, w, 4]
    y: dict
        the dict of output yuv422, shape is [h, w, 2]
    kernel_name: str
        cce kernel name, default value is "yuv444_2_yuv422"

    Returns
    -------
    tik instance
    """
    _check_input_params(x, y)

    obj = Yuv4442Yuv422(x, y)
    obj.yuv444_2_yuv422_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num
    })

    tik_inst = obj.tik_inst
    inputs_gm, outputs_gm = obj.get_inputs_outputs_gm()
    opt_config = {"enable_const_fold": True}
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=inputs_gm,
                      outputs=outputs_gm,
                      flowtable=(obj.tiling_gm,),
                      config=opt_config)

    return tik_inst
