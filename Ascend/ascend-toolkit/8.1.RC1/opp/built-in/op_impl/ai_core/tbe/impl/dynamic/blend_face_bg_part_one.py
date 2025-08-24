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
blend_face_bg_part_one
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_common
from impl.constant_util_v1 import SCALAR_MIN_FP32


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
class BlendFaceBg():
    """
    BlendFaceBg class
    """
    MAX_INT32 = 2 ** 31 - 1
    MASK_F32 = 64
    MASK_F16 = 128
    BLOCK_I64 = 4
    BLOCK_F32 = 8
    BLOCK_U8 = 32

    ROW_SLICE = 2048
    TILING_ARG_NUM = 8

    def __init__(self, face_img, acc_face):
        """
        init
        """
        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.c = 3
        self.face_img_dtype = face_img.get("dtype").lower()
        self.dtype_f32 = acc_face.get("dtype").lower()
        self.dtype_f16 = "float16"
        self.dtype_i32 = "int32"
        self.tiling_dtype = "int64"
        self.tiling_align = util_common.align(self.TILING_ARG_NUM, self.BLOCK_I64)

        self.face_img_gm = None
        self.face_rect_gm = None
        self.face_mask_gm = None
        self.acc_face_gm = None
        self.acc_mask_gm = None
        self.max_mask_gm = None
        self.acc_face_out_gm = None
        self.acc_mask_out_gm = None
        self.max_mask_out_gm = None
        self.tiling_gm = None

        self._init_gm_tensor()

        # tiling params
        self.h = None
        self.w = None
        self.h_out = None
        self.w_out = None
        self.need_core_num = None
        self.low_core_num = None
        self.rows_num_low = None

        # rect coordinate
        self.left = None
        self.top = None
        self.right = None
        self.bottom = None

    def blend_face_bg_compute(self):
        """
        blend_face_bg part one compute
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            # get tiling data
            self._get_tiling_args()

            with self.tik_inst.if_scope(core_id < self.need_core_num):
                self._get_rect_data()

                with self.tik_inst.if_scope(core_id < self.low_core_num):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.rows_num_low * core_id
                        self._one_core_compute(start_idx, self.rows_num_low)

                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        rows_num = self.rows_num_low - 1
                        start_idx = self.rows_num_low * self.low_core_num + rows_num * (core_id - self.low_core_num)
                        self._one_core_compute(start_idx, rows_num)

    def get_inputs_outputs_gm(self):
        """
        inputs and outputs gm tensor
        """
        inputs_gm = (self.face_img_gm, self.face_rect_gm, self.face_mask_gm, self.acc_face_gm,
                     self.acc_mask_gm, self.max_mask_gm)
        outputs_gm = (self.acc_face_out_gm, self.acc_mask_out_gm, self.max_mask_out_gm)

        return inputs_gm, outputs_gm

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        self.face_img_gm = self.tik_inst.Tensor(self.face_img_dtype, (self.MAX_INT32,), name="face_img_gm",
                                                scope=tik.scope_gm)
        self.face_rect_gm = self.tik_inst.Tensor(self.dtype_i32, (4,), name="face_rect_gm", scope=tik.scope_gm)
        self.face_mask_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="face_mask_gm",
                                                 scope=tik.scope_gm)
        self.acc_face_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="acc_face_gm",
                                                scope=tik.scope_gm)
        self.acc_mask_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="acc_mask_gm",
                                                scope=tik.scope_gm)
        self.max_mask_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="max_mask_gm",
                                                scope=tik.scope_gm)

        self.acc_face_out_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="acc_face_out_gm",
                                                    scope=tik.scope_gm)
        self.acc_mask_out_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="acc_mask_out_gm",
                                                    scope=tik.scope_gm)
        self.max_mask_out_gm = self.tik_inst.Tensor(self.dtype_f32, (self.MAX_INT32,), name="max_mask_out_gm",
                                                    scope=tik.scope_gm)

        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.h = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h")
        self.w = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w")
        self.h_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h_out")
        self.w_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_out")
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.low_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="low_core_num")
        self.rows_num_low = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="rows_num_low")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.BLOCK_I64, 0, 0)

            self.h.set_as(tiling_ub[0])
            self.w.set_as(tiling_ub[1])
            self.h_out.set_as(tiling_ub[2])
            self.w_out.set_as(tiling_ub[3])
            self.need_core_num.set_as(tiling_ub[4])
            self.low_core_num.set_as(tiling_ub[5])
            self.rows_num_low.set_as(tiling_ub[6])

    def _get_rect_data(self):
        """
        get face rect data
        """
        self.left = self.tik_inst.Scalar(dtype=self.dtype_i32, name="left")
        self.top = self.tik_inst.Scalar(dtype=self.dtype_i32, name="top")
        self.right = self.tik_inst.Scalar(dtype=self.dtype_i32, name="right")
        self.bottom = self.tik_inst.Scalar(dtype=self.dtype_i32, name="bottom")

        with self.tik_inst.new_stmt_scope():
            face_rect_ub = self.tik_inst.Tensor(self.dtype_i32, (self.BLOCK_F32,), name="face_rect_ub",
                                                scope=tik.scope_ubuf)
            self.tik_inst.data_move(face_rect_ub, self.face_rect_gm, 0, 1, 1, 0, 0)

            self.left.set_as(face_rect_ub[0])
            self.top.set_as(face_rect_ub[1])
            self.right.set_as(face_rect_ub[2])
            self.bottom.set_as(face_rect_ub[3])

    def _face_mask_trans_first(self, face_mask_tmp_ub, base_offset, repeats, w_elems):
        with self.tik_inst.new_stmt_scope():
            face_mask_ori_ub = self.tik_inst.Tensor(self.dtype_f32, (self.ROW_SLICE,),
                                                    name="face_mask_ori_ub", scope=tik.scope_ubuf)
            self.tik_inst.data_move(face_mask_ori_ub, self.face_mask_gm[base_offset], 0, 1,
                                    util_common.ceil(w_elems, self.BLOCK_F32), 0, 0)

            index_list = []
            for i in range(8):
                index_list.extend([16 * repeats * i, 8 + 16 * repeats * i])

            src_list_1 = [face_mask_ori_ub[16 * i] for i in range(16)]
            dst_list_1 = [face_mask_tmp_ub[i] for i in index_list]

            src_list_2 = [face_mask_ori_ub[16 * i + 8] for i in range(16)]
            dst_list_2 = [face_mask_tmp_ub[i + 128 * repeats] for i in index_list]

            with self.tik_inst.if_scope(repeats == 1):
                self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
                self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
            with self.tik_inst.else_scope():
                self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeats, 16 // 8, 256 // 8)
                self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeats, 16 // 8, 256 // 8)

    def _face_mask_trans(self, face_mask_ub, base_offset, repeats, w_elems):
        """
        convert [w,1] to [w,3]
        """
        with self.tik_inst.new_stmt_scope():
            face_mask_tmp_ub = self.tik_inst.Tensor(self.dtype_f32, (self.ROW_SLICE,),
                                                    name="face_mask_tmp_ub", scope=tik.scope_ubuf)
            self._face_mask_trans_first(face_mask_tmp_ub, base_offset, repeats, w_elems)

            repeat2 = repeats * 2
            dst_rep_stride = 128 * self.c // 8
            src_rep_stride = 8 // 8

            index_list = []
            for i in range(8):
                index_list.extend([48 * i, 8 + 48 * i])

            list_1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]
            src_list_1 = [face_mask_tmp_ub[8 * repeat2 * i] for i in list_1]
            dst_list_1 = [face_mask_ub[i] for i in index_list]

            list_2 = [5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]
            src_list_2 = [face_mask_tmp_ub[8 * repeat2 * i] for i in list_2]
            dst_list_2 = [face_mask_ub[i + 16] for i in index_list]

            list_3 = [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15]
            src_list_3 = [face_mask_tmp_ub[8 * repeat2 * i] for i in list_3]
            dst_list_3 = [face_mask_ub[i + 32] for i in index_list]

            with self.tik_inst.new_stmt_scope():
                self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2, dst_rep_stride, src_rep_stride)
                self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2, dst_rep_stride, src_rep_stride)
                self.tik_inst.vnchwconv(False, False, dst_list_3, src_list_3, repeat2, dst_rep_stride, src_rep_stride)

    def _cal_acc_face(self, face_mask_ub, base_offset, out_offset, elems):
        """
        calculate output 0, acc_face
        """
        with self.tik_inst.new_stmt_scope():
            face_img_ub = self.tik_inst.Tensor(self.dtype_f32, (self.ROW_SLICE * self.c,),
                                               name="face_img_ub", scope=tik.scope_ubuf)
            burst_len = util_common.ceil(elems, self.BLOCK_F32)
            face_offset = base_offset * self.c

            if self.face_img_dtype == "uint8":
                with self.tik_inst.new_stmt_scope():
                    face_img_u8 = self.tik_inst.Tensor(self.face_img_dtype, (self.ROW_SLICE * self.c,),
                                                       name="face_img_u8", scope=tik.scope_ubuf)
                    face_img_f16 = self.tik_inst.Tensor(self.dtype_f16, (self.ROW_SLICE * self.c,),
                                                        name="face_img_f16", scope=tik.scope_ubuf)
                    self.tik_inst.data_move(face_img_u8, self.face_img_gm[face_offset], 0, 1,
                                            util_common.ceil(elems, self.BLOCK_U8), 0, 0)
                    self.tik_inst.vconv(self.MASK_F16, "", face_img_f16, face_img_u8,
                                        util_common.ceil(elems, self.MASK_F16), 1, 1, 8, 4)
                    self.tik_inst.vconv(self.MASK_F32, "", face_img_ub, face_img_f16,
                                        util_common.ceil(elems, self.MASK_F32), 1, 1, 8, 4)
            else:
                self.tik_inst.data_move(face_img_ub, self.face_img_gm[face_offset], 0, 1, burst_len, 0, 0)

            self.tik_inst.vmul(self.MASK_F32, face_img_ub, face_img_ub, face_mask_ub,
                               util_common.ceil(elems, self.MASK_F32), 1, 1, 1, 8, 8, 8)

            self.tik_inst.set_atomic_add(1)
            self.tik_inst.data_move(self.acc_face_out_gm[out_offset], face_img_ub, 0, 1, burst_len, 0, 0)
            self.tik_inst.set_atomic_add(0)

    def _cal_acc_mask(self, face_mask_ub, out_offset, elems):
        """
        calculate output 1, acc_mask
        """
        self.tik_inst.set_atomic_add(1)
        self.tik_inst.data_move(self.acc_mask_out_gm[out_offset], face_mask_ub, 0, 1,
                                util_common.ceil(elems, self.BLOCK_F32), 0, 0)
        self.tik_inst.set_atomic_add(0)

    def _cal_max_mask(self, face_mask_ub, out_offset, elems):
        """
        calculate output 2, max_mask
        """
        with self.tik_inst.new_stmt_scope():
            max_mask_ub = self.tik_inst.Tensor(self.dtype_f32, (self.ROW_SLICE * self.c,),
                                               name="max_mask_ub", scope=tik.scope_ubuf)
            burst_len = util_common.ceil(elems, self.BLOCK_F32)
            self.tik_inst.data_move(max_mask_ub, self.max_mask_gm[out_offset], 0, 1, burst_len, 0, 0)

            self.tik_inst.vmax(self.MASK_F32, max_mask_ub, max_mask_ub, face_mask_ub,
                               util_common.ceil(elems, self.MASK_F32), 1, 1, 1, 8, 8, 8)

            self.tik_inst.data_move(self.max_mask_out_gm[out_offset], max_mask_ub, 0, 1, burst_len, 0, 0)

    def _one_row_process(self, rows_idx, params):
        """
        process one row.
        params is (w_loop, w_tail, repeats, repeats_tail)
        """
        (w_loop, w_tail, repeats, repeats_tail) = params

        with self.tik_inst.for_range(0, w_loop) as loop_i:
            with self.tik_inst.new_stmt_scope():
                base_offset = rows_idx * self.w + self.ROW_SLICE * loop_i
                out_offset = ((rows_idx + self.top) * self.w_out + self.left + self.ROW_SLICE * loop_i) * self.c
                face_mask_ub = self.tik_inst.Tensor(self.dtype_f32, (self.ROW_SLICE * self.c,),
                                                    name="face_mask_ub", scope=tik.scope_ubuf)
                self._face_mask_trans(face_mask_ub, base_offset, repeats, self.ROW_SLICE)

                self._cal_acc_face(face_mask_ub, base_offset, out_offset, self.ROW_SLICE * self.c)
                self._cal_acc_mask(face_mask_ub, out_offset, self.ROW_SLICE * self.c)
                self._cal_max_mask(face_mask_ub, out_offset, self.ROW_SLICE * self.c)

        with self.tik_inst.if_scope(w_tail > 0):
            with self.tik_inst.new_stmt_scope():
                base_offset = rows_idx * self.w + self.ROW_SLICE * w_loop
                out_offset = ((rows_idx + self.top) * self.w_out + self.left + self.ROW_SLICE * w_loop) * self.c
                face_mask_ub = self.tik_inst.Tensor(self.dtype_f32, (self.ROW_SLICE * self.c,),
                                                    name="face_mask_ub", scope=tik.scope_ubuf)
                self._face_mask_trans(face_mask_ub, base_offset, repeats_tail, w_tail)

                elems = w_tail * self.c
                # filled with 0
                with self.tik_inst.if_scope(elems % self.BLOCK_F32 != 0):
                    for i in range(self.BLOCK_F32 - 1):
                        face_mask_ub[elems + i].set_as(0)
                self._cal_acc_face(face_mask_ub, base_offset, out_offset, elems)
                self._cal_acc_mask(face_mask_ub, out_offset, elems)

                # filled with min value of float32
                with self.tik_inst.if_scope(elems % self.BLOCK_F32 != 0):
                    for i in range(self.BLOCK_F32 - 1):
                        face_mask_ub[elems + i].set_as(SCALAR_MIN_FP32)
                self._cal_max_mask(face_mask_ub, out_offset, elems)

    def _one_core_compute(self, rows_start_idx, rows_num):
        """
        compute for one core
        """
        w_loop = self.w // self.ROW_SLICE
        w_tail = self.w % self.ROW_SLICE
        
        repeats = self.ROW_SLICE // 256
        repeats_tail = util_common.ceil(w_tail, 256)

        with self.tik_inst.for_range(0, rows_num) as idx:
            self._one_row_process(rows_start_idx + idx, (w_loop, w_tail, repeats, repeats_tail))


def _check_input_params(input_list, output_list):
    """
    check input parameters.
    input_list is (face_img, face_rect, face_mask, acc_face, acc_mask, max_mask)
    output_list is (acc_face_out, acc_mask_out, max_mask_out)
    """
    (face_img, face_rect, face_mask, acc_face, acc_mask, max_mask) = input_list
    (acc_face_out, acc_mask_out, max_mask_out) = output_list

    face_img_dtype = face_img.get("dtype").lower()
    face_rect_dtype = face_rect.get("dtype").lower()
    face_mask_dtype = face_mask.get("dtype").lower()
    acc_face_dtype = acc_face.get("dtype").lower()
    acc_mask_dtype = acc_mask.get("dtype").lower()
    max_mask_dtype = max_mask.get("dtype").lower()
    acc_face_out_dtype = acc_face_out.get("dtype").lower()
    acc_mask_out_dtype = acc_mask_out.get("dtype").lower()
    max_mask_out_dtype = max_mask_out.get("dtype").lower()
    para_check.check_dtype(face_img_dtype, ("uint8", "float32"), param_name="face_img")
    para_check.check_dtype(face_rect_dtype, ("int32",), param_name="face_rect")
    para_check.check_dtype(face_mask_dtype, ("float32",), param_name="face_mask")
    para_check.check_dtype(acc_face_dtype, ("float32",), param_name="acc_face")
    para_check.check_dtype(acc_mask_dtype, ("float32",), param_name="acc_mask")
    para_check.check_dtype(max_mask_dtype, ("float32",), param_name="max_mask")
    para_check.check_dtype(acc_face_out_dtype, ("float32",), param_name="acc_face_out")
    para_check.check_dtype(acc_mask_out_dtype, ("float32",), param_name="acc_mask_out")
    para_check.check_dtype(max_mask_out_dtype, ("float32",), param_name="max_mask_out")

    face_rect_shape = face_rect.get("shape")
    para_check.check_shape(face_rect_shape, min_rank=1, max_rank=1, param_name="face_rect")


@register_operator("BlendFaceBgPartOne")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def blend_face_bg_part_one(face_img, face_rect, face_mask, acc_face, acc_mask, max_mask,
                           acc_face_out, acc_mask_out, max_mask_out, kernel_name="blend_face_bg_part_one"):
    """
    BlendFaceBgPartOne op

    Parameters
    ----------
    face_img: dict
        the dict of input face_img, shape is [h, w, 3]
    face_rect: dict
        the dict of input face_rect, shape is [4], value is [left, top, right, bottom]
    face_mask: dict
        the dict of input face_mask, shape is [h, w, 1]
    acc_face: dict
        the dict of input acc_face, shape is [H, W, 3]
    acc_mask: dict
        the dict of input acc_mask, shape is [H, W, 3]
    max_mask: dict
        the dict of input max_mask, shape is [H, W, 3]
    acc_face_out: dict
        the dict of output acc_face_out, shape is [H, W, 3]
    acc_mask_out: dict
        the dict of output acc_mask_out, shape is [H, W, 3]
    max_mask_out: dict
        the dict of output max_mask_out, shape is [H, W, 3]
    kernel_name: str
        cce kernel name, default value is "blend_face_bg_part_one"

    Returns
    -------
    tik instance
    """
    input_list = (face_img, face_rect, face_mask, acc_face, acc_mask, max_mask)
    output_list = (acc_face_out, acc_mask_out, max_mask_out)
    _check_input_params(input_list, output_list)

    obj = BlendFaceBg(face_img, acc_face)
    obj.blend_face_bg_compute()

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
