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
aipp
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_SHAPE_SIZE = 2 ** 32 - 1
    # bytes of one block
    BLOCK_BYTES = 32
    # tiling param num
    TILING_ARG_NUM = 8
    PARAM_HEAD_STRUCT_SIZE = 64
    PARAM_BATCH_STRUCT_SIZE = 96
    # uint8
    HEAD_OFFSET_INPUT_FORMAT = 0
    HEAD_OFFSET_SRC_IMAGE_SIZE_W = 8
    HEAD_OFFSET_CSC_MATRIX_R0C0 = 16
    HEAD_OFFSET_CROP_START_W = 8
    HEAD_OFFSET_DTC_MEAN_CHN0 = 56
    HEAD_OFFSET_DTC_MIN_CHN0 = 64
    # aipp input format
    FORMAT_YUV420SP = 1
    FORMAT_XRGB8888 = 2
    FORMAT_RGB888 = 5
    FORMAT_ARGB8888 = 6
    FORMAT_YUYV = 7
    FORMAT_YUV422SP = 8
    FORMAT_AYUV444 = 9
    FORMAT_YUV400 = 10


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


def align_value(value, factor):
    """
    Alignment based on factor.

    Parameters
    ----------
    value: input number
    factor: alignment base

    Returns
    -------
    aligned value
    """
    return (value + factor - 1) // factor * factor


# 'pylint: disable=invalid-name
def raise_runtime_error(cause_desc):
    """
    raise runtime error
    """
    aipp_op_error_code = 'E81012'
    error_info = {'errCode': aipp_op_error_code, 'cause_desc': cause_desc}

    raise RuntimeError(error_info, "Compile op[aipp] failed, cause: %s." % cause_desc)


def check_input_params(input_data, input_dync_param, output_data):
    """
    check input params
    """
    if (not input_dync_param) or ("dtype" not in input_dync_param):
        cause_desc = "aipp dynamic shape, input params is invalid"
        raise_runtime_error(cause_desc)

    data_dtype = input_data.get('dtype').lower()
    data_format = input_data.get('format')
    input_dync_param_dtype = input_dync_param.get('dtype').lower()
    output_dtype = output_data.get('dtype').lower()
    output_format = output_data.get('format')
    cur_cce_product = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)

    para_check.check_dtype(data_dtype, ("uint8",), param_name="input_dtype")
    para_check.check_dtype(input_dync_param_dtype, ("uint8",), param_name="input_dync_param_dtype")
    para_check.check_dtype(output_dtype, ("float16", "uint8", "int8"), param_name="output")

    input_format_list = ("NCHW", "NHWC")
    para_check.check_format(data_format, input_format_list, param_name="input_data")

    if cur_cce_product not in (tbe_platform.ASCEND_310, tbe_platform.ASCEND_910,
                               tbe_platform.ASCEND_310P):
        cause_desc = "aipp dynamic shape only support Ascend310, Ascend910, Ascend310P"
        raise_runtime_error(cause_desc)

    para_check.check_format(output_format, ("NC1HWC0",), param_name="output_data")


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments,unused-argument
# 'pylint: disable=too-many-instance-attributes,too-many-public-methods,too-many-lines
class Aipp():
    """
    Function: class that execute Aipp
    """
    def __init__(self, input_data, input_dync_param, output_data):
        """
        Aipp init
        """
        # reserved ub size
        reserved_ub_size = 2 * 1024
        # 8 bit
        eight_bit = 8
        # padding to 256bits
        channel_pad_mode_0 = 0
        self.tik_instance = tik.Tik()
        self.input_dtype = input_data.get('dtype').lower()
        self.data_format = input_data.get('format')
        self.param_dtype = input_dync_param.get('dtype').lower()
        self.output_dtype = output_data.get('dtype').lower()
        self.output_format = output_data.get('format')
        self.output_dsize = get_bit_len(self.output_dtype) // eight_bit
        self.block_elems = Constant.BLOCK_BYTES // self.output_dsize

        self.c0 = 16
        self.c_padding_value_zero = 0.0
        self.area_padding_value_zero = 0.0
        self.channel_pad_mode = channel_pad_mode_0
        if self.output_dtype != "float16":
            self.c0 = 32
            self.c_padding_value_zero = 0
            self.area_padding_value_zero = 0

        self.cur_cce_product = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - reserved_ub_size
        self.l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        self.ihisi = False
        if self.cur_cce_product in (tbe_platform.HI3796CV300ES, tbe_platform.HI3796CV300CS, tbe_platform.SD3403):
            self.ihisi = True
        self.ub_max_elems = (self.ub_size // self.output_dsize) // self.block_elems * self.block_elems
        self.max_s8_padding_elems = (self.ub_size // 3) // Constant.BLOCK_BYTES * Constant.BLOCK_BYTES
        self.l1_max_elems = (self.l1_size // self.output_dsize) // self.block_elems * self.block_elems
        self.max_hw_l1_size = self.l1_max_elems // self.c0

        self.tiling_dtype = "int64"
        self.tiling_align = align_value(Constant.TILING_ARG_NUM, 4)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.data_gm = self.tik_instance.Tensor(self.input_dtype, (Constant.MAX_SHAPE_SIZE,),
                                                name="input_data", scope=tik.scope_gm)
        self.dync_param_gm = self.tik_instance.Tensor(self.param_dtype, (Constant.MAX_SHAPE_SIZE,),
                                                      name="input_dync_param", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.output_dtype, (Constant.MAX_SHAPE_SIZE,),
                                                  name="output_data", scope=tik.scope_gm)

        self.tiling_ub = None
        self.params_ub = None
        self.batch_params_ub = None

        self.csc_matrix = None
        self.csc_out_bias = None
        self.csc_in_bias = None
        self.swap_list = None
        self.dtc_mean_list = None
        self.dtc_min_list = None
        self.dtc_reci_list = None

        # head params
        self.input_format = None
        self.output_bias_r0 = None
        self.output_bias_r1 = None
        self.output_bias_r2 = None
        self.input_bias_r0 = None
        self.input_bias_r1 = None
        self.input_bias_r2 = None
        self.csc_switch = None
        self.rbuv_swap_switch = None
        self.ax_swap_switch = None
        self.batch = None
        self.src_image_size_h = None
        self.src_image_size_w = None
        self.matrix_r0_c0 = None
        self.matrix_r0_c1 = None
        self.matrix_r0_c2 = None
        self.matrix_r1_c0 = None
        self.matrix_r1_c1 = None
        self.matrix_r1_c2 = None
        self.matrix_r2_c0 = None
        self.matrix_r2_c1 = None
        self.matrix_r2_c2 = None

        # batch params
        self.crop_switch = None
        self.scf_switch = None
        self.padding_switch = None

        self.crop_start_pos_w = None
        self.crop_start_pos_h = None
        self.crop_size_w = None
        self.crop_size_h = None

        self.scf_input_size_w = None
        self.scf_input_size_h = None
        self.scf_output_size_w = None
        self.scf_output_size_h = None

        self.padding_size_top = None
        self.padding_size_bottom = None
        self.padding_size_left = None
        self.padding_size_right = None

        self.dtc_mean_chn0 = None
        self.dtc_mean_chn1 = None
        self.dtc_mean_chn2 = None
        self.dtc_mean_chn3 = None
        self.dtc_min_chn0 = None
        self.dtc_min_chn1 = None
        self.dtc_min_chn2 = None
        self.dtc_min_chn3 = None
        self.dtc_reci_chn0 = None
        self.dtc_reci_chn1 = None
        self.dtc_reci_chn2 = None
        self.dtc_reci_chn3 = None

        # tiling params
        self.need_core_num = None
        self.output_n = None
        self.output_c1 = None
        self.output_h = None
        self.output_w = None
        self.output_c0 = None
        self.batch_each_core = None
        self.batch_last_core = None

    def get_tiling_args(self):
        """
        get runtime params from tiling data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.output_n = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="output_n")
        self.output_c1 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="output_c1")
        self.output_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="output_h")
        self.output_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="output_w")
        self.output_c0 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="output_c0")
        self.batch_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="batch_each_core")
        self.batch_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="batch_last_core")

        self.need_core_num.set_as(self.tiling_ub[0])
        self.output_n.set_as(self.tiling_ub[1])
        self.output_c1.set_as(self.tiling_ub[2])
        self.output_h.set_as(self.tiling_ub[3])
        self.output_w.set_as(self.tiling_ub[4])
        self.output_c0.set_as(self.tiling_ub[5])
        self.batch_each_core.set_as(self.tiling_ub[6])
        self.batch_last_core.set_as(self.tiling_ub[7])

    def init_batch_params(self):
        """
        batch params definition
        """
        self.crop_switch = self.tik_instance.Scalar(dtype="int8", name="crop_switch")
        self.scf_switch = self.tik_instance.Scalar(dtype="int8", name="scf_switch")
        self.padding_switch = self.tik_instance.Scalar(dtype="int8", name="padding_switch")

        self.crop_start_pos_w = self.tik_instance.Scalar(dtype="int32", name="crop_start_pos_w")
        self.crop_start_pos_h = self.tik_instance.Scalar(dtype="int32", name="crop_start_pos_h")
        self.crop_size_w = self.tik_instance.Scalar(dtype="int32", name="crop_size_w")
        self.crop_size_h = self.tik_instance.Scalar(dtype="int32", name="crop_size_h")

        self.scf_input_size_w = self.tik_instance.Scalar(dtype="int32", name="scf_input_size_w", init_value=0)
        self.scf_input_size_h = self.tik_instance.Scalar(dtype="int32", name="scf_input_size_h", init_value=0)
        self.scf_output_size_w = self.tik_instance.Scalar(dtype="int32", name="scf_output_size_w", init_value=0)
        self.scf_output_size_h = self.tik_instance.Scalar(dtype="int32", name="scf_output_size_h", init_value=0)

        self.padding_size_top = self.tik_instance.Scalar(dtype="int32", name="padding_size_top", init_value=0)
        self.padding_size_bottom = self.tik_instance.Scalar(dtype="int32", name="padding_size_bottom", init_value=0)
        self.padding_size_left = self.tik_instance.Scalar(dtype="int32", name="padding_size_left", init_value=0)
        self.padding_size_right = self.tik_instance.Scalar(dtype="int32", name="padding_size_right", init_value=0)

        self.dtc_mean_chn0 = self.tik_instance.Scalar(dtype="int16", name="dtc_mean_chn0")
        self.dtc_mean_chn1 = self.tik_instance.Scalar(dtype="int16", name="dtc_mean_chn1")
        self.dtc_mean_chn2 = self.tik_instance.Scalar(dtype="int16", name="dtc_mean_chn2")
        self.dtc_mean_chn3 = self.tik_instance.Scalar(dtype="int16", name="dtc_mean_chn3")

        self.dtc_min_chn0 = self.tik_instance.Scalar(dtype="float16", name="dtc_min_chn0")
        self.dtc_min_chn1 = self.tik_instance.Scalar(dtype="float16", name="dtc_min_chn1")
        self.dtc_min_chn2 = self.tik_instance.Scalar(dtype="float16", name="dtc_min_chn2")
        self.dtc_min_chn3 = self.tik_instance.Scalar(dtype="float16", name="dtc_min_chn3")

        self.dtc_reci_chn0 = self.tik_instance.Scalar(dtype="float16", name="dtc_reci_chn0")
        self.dtc_reci_chn1 = self.tik_instance.Scalar(dtype="float16", name="dtc_reci_chn1")
        self.dtc_reci_chn2 = self.tik_instance.Scalar(dtype="float16", name="dtc_reci_chn2")
        self.dtc_reci_chn3 = self.tik_instance.Scalar(dtype="float16", name="dtc_reci_chn3")

    def get_src_image_size(self):
        """
        get src image size height and width
        """
        self.src_image_size_h = self.tik_instance.Scalar(dtype="int32", name="src_image_size_h")
        self.src_image_size_w = self.tik_instance.Scalar(dtype="int32", name="src_image_size_w")
        with self.tik_instance.new_stmt_scope():
            src_image_size_ub = self.params_ub[Constant.HEAD_OFFSET_SRC_IMAGE_SIZE_W:\
            Constant.HEAD_OFFSET_CSC_MATRIX_R0C0].reinterpret_cast_to("int32")

            self.src_image_size_w.set_as(src_image_size_ub[0])
            self.src_image_size_h.set_as(src_image_size_ub[1])

    def get_csc_matrix(self):
        """
        get csc matrix
        """
        head_offset_csc_matrix_r2r2_end = 34
        self.matrix_r0_c0 = self.tik_instance.Scalar(dtype="int16", name="matrix_r0_c0")
        self.matrix_r0_c1 = self.tik_instance.Scalar(dtype="int16", name="matrix_r0_c1")
        self.matrix_r0_c2 = self.tik_instance.Scalar(dtype="int16", name="matrix_r0_c2")
        self.matrix_r1_c0 = self.tik_instance.Scalar(dtype="int16", name="matrix_r1_c0")
        self.matrix_r1_c1 = self.tik_instance.Scalar(dtype="int16", name="matrix_r1_c1")
        self.matrix_r1_c2 = self.tik_instance.Scalar(dtype="int16", name="matrix_r1_c2")
        self.matrix_r2_c0 = self.tik_instance.Scalar(dtype="int16", name="matrix_r2_c0")
        self.matrix_r2_c1 = self.tik_instance.Scalar(dtype="int16", name="matrix_r2_c1")
        self.matrix_r2_c2 = self.tik_instance.Scalar(dtype="int16", name="matrix_r2_c2")

        with self.tik_instance.new_stmt_scope():
            csc_matrix_ub = self.params_ub[Constant.HEAD_OFFSET_CSC_MATRIX_R0C0:head_offset_csc_matrix_r2r2_end]\
            .reinterpret_cast_to("int16")

            self.matrix_r0_c0.set_as(csc_matrix_ub[0])
            self.matrix_r0_c1.set_as(csc_matrix_ub[1])
            self.matrix_r0_c2.set_as(csc_matrix_ub[2])
            self.matrix_r1_c0.set_as(csc_matrix_ub[3])
            self.matrix_r1_c1.set_as(csc_matrix_ub[4])
            self.matrix_r1_c2.set_as(csc_matrix_ub[5])
            self.matrix_r2_c0.set_as(csc_matrix_ub[6])
            self.matrix_r2_c1.set_as(csc_matrix_ub[7])
            self.matrix_r2_c2.set_as(csc_matrix_ub[8])

    def get_public_params(self):
        """
        get public params
        """
        # int8
        head_offset_csc_switch = 1
        head_offset_rbuv_swap_switch = 2
        head_offset_ax_swap_switch = 3
        head_offset_batch_num = 4
        # uint8
        head_offset_csc_output_bias_r0 = 40
        head_offset_csc_output_bias_r1 = 41
        head_offset_csc_output_bias_r2 = 42
        head_offset_csc_input_bias_r0 = 43
        head_offset_csc_input_bias_r1 = 44
        head_offset_csc_input_bias_r2 = 45
        # move data of head params to ub, uint8
        self.params_ub = self.tik_instance.Tensor(self.param_dtype, (Constant.PARAM_HEAD_STRUCT_SIZE,), \
        name="params_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.params_ub, self.dync_param_gm, 0, 1, Constant.PARAM_HEAD_STRUCT_SIZE // \
        Constant.BLOCK_BYTES, 0, 0)

        self.input_format = self.tik_instance.Scalar(dtype="uint8", name="input_format")
        self.input_format.set_as(self.params_ub[Constant.HEAD_OFFSET_INPUT_FORMAT])

        self.output_bias_r0 = self.tik_instance.Scalar(dtype="uint8", name="output_bias_r0")
        self.output_bias_r1 = self.tik_instance.Scalar(dtype="uint8", name="output_bias_r1")
        self.output_bias_r2 = self.tik_instance.Scalar(dtype="uint8", name="output_bias_r2")
        self.input_bias_r0 = self.tik_instance.Scalar(dtype="uint8", name="input_bias_r0")
        self.input_bias_r1 = self.tik_instance.Scalar(dtype="uint8", name="input_bias_r1")
        self.input_bias_r2 = self.tik_instance.Scalar(dtype="uint8", name="input_bias_r2")
        self.output_bias_r0.set_as(self.params_ub[head_offset_csc_output_bias_r0])
        self.output_bias_r1.set_as(self.params_ub[head_offset_csc_output_bias_r1])
        self.output_bias_r2.set_as(self.params_ub[head_offset_csc_output_bias_r2])
        self.input_bias_r0.set_as(self.params_ub[head_offset_csc_input_bias_r0])
        self.input_bias_r1.set_as(self.params_ub[head_offset_csc_input_bias_r1])
        self.input_bias_r2.set_as(self.params_ub[head_offset_csc_input_bias_r2])

        self.csc_switch = self.tik_instance.Scalar(dtype="int8", name="csc_switch")
        self.rbuv_swap_switch = self.tik_instance.Scalar(dtype="int8", name="rbuv_swap_switch")
        self.ax_swap_switch = self.tik_instance.Scalar(dtype="int8", name="ax_swap_switch")
        self.batch = self.tik_instance.Scalar(dtype="int8", name="batch")

        with self.tik_instance.new_stmt_scope():
            public_switch_ub = \
                self.params_ub[Constant.HEAD_OFFSET_INPUT_FORMAT:Constant.HEAD_OFFSET_SRC_IMAGE_SIZE_W].\
                reinterpret_cast_to("int8")

            self.csc_switch.set_as(public_switch_ub[head_offset_csc_switch])
            self.rbuv_swap_switch.set_as(public_switch_ub[head_offset_rbuv_swap_switch])
            self.ax_swap_switch.set_as(public_switch_ub[head_offset_ax_swap_switch])
            self.batch.set_as(public_switch_ub[head_offset_batch_num])

        self.get_src_image_size()
        # csc matrix
        self.get_csc_matrix()

    def get_dtc_params(self):
        """
        get dtc params
        """
        head_offset_dtc_reci_chn3_end = 80
        with self.tik_instance.new_stmt_scope():
            batch_dtc_mean_ub = self.batch_params_ub[Constant.HEAD_OFFSET_DTC_MEAN_CHN0:\
            Constant.HEAD_OFFSET_DTC_MIN_CHN0].reinterpret_cast_to("int16")
            batch_dtc_min_reci_ub = self.batch_params_ub[Constant.HEAD_OFFSET_DTC_MIN_CHN0:\
            head_offset_dtc_reci_chn3_end].reinterpret_cast_to("float16")

            self.dtc_mean_chn0.set_as(batch_dtc_mean_ub[0])
            self.dtc_mean_chn1.set_as(batch_dtc_mean_ub[1])
            self.dtc_mean_chn2.set_as(batch_dtc_mean_ub[2])
            self.dtc_mean_chn3.set_as(batch_dtc_mean_ub[3])

            self.dtc_min_chn0.set_as(batch_dtc_min_reci_ub[0])
            self.dtc_min_chn1.set_as(batch_dtc_min_reci_ub[1])
            self.dtc_min_chn2.set_as(batch_dtc_min_reci_ub[2])
            self.dtc_min_chn3.set_as(batch_dtc_min_reci_ub[3])

            self.dtc_reci_chn0.set_as(batch_dtc_min_reci_ub[4])
            self.dtc_reci_chn1.set_as(batch_dtc_min_reci_ub[5])
            self.dtc_reci_chn2.set_as(batch_dtc_min_reci_ub[6])
            self.dtc_reci_chn3.set_as(batch_dtc_min_reci_ub[7])

    def get_crop_scf_padding_params(self):
        """
        get crop, scf, padding params
        """
        with self.tik_instance.new_stmt_scope():
            batch_crop_scf_padding_ub = self.batch_params_ub[Constant.HEAD_OFFSET_CROP_START_W:\
            Constant.HEAD_OFFSET_DTC_MEAN_CHN0].reinterpret_cast_to("int32")

            with self.tik_instance.if_scope(self.crop_switch > 0):
                self.crop_start_pos_w.set_as(batch_crop_scf_padding_ub[0])
                self.crop_start_pos_h.set_as(batch_crop_scf_padding_ub[1])
                self.crop_size_w.set_as(batch_crop_scf_padding_ub[2])
                self.crop_size_h.set_as(batch_crop_scf_padding_ub[3])
            with self.tik_instance.else_scope():
                self.crop_start_pos_w.set_as(0)
                self.crop_start_pos_h.set_as(0)
                self.crop_size_w.set_as(self.src_image_size_w)
                self.crop_size_h.set_as(self.src_image_size_h)

            if self.ihisi:
                with self.tik_instance.if_scope(self.scf_switch > 0):
                    self.scf_input_size_w.set_as(batch_crop_scf_padding_ub[4])
                    self.scf_input_size_h.set_as(batch_crop_scf_padding_ub[5])
                    with self.tik_instance.if_scope(self.scf_input_size_w == 0):
                        self.scf_input_size_w.set_as(self.crop_size_w)
                    with self.tik_instance.if_scope(self.scf_input_size_h == 0):
                        self.scf_input_size_h.set_as(self.crop_size_h)

                    self.scf_output_size_w.set_as(batch_crop_scf_padding_ub[6])
                    self.scf_output_size_h.set_as(batch_crop_scf_padding_ub[7])

            with self.tik_instance.if_scope(self.padding_switch > 0):
                self.padding_size_top.set_as(batch_crop_scf_padding_ub[8])
                self.padding_size_bottom.set_as(batch_crop_scf_padding_ub[9])
                self.padding_size_left.set_as(batch_crop_scf_padding_ub[10])
                self.padding_size_right.set_as(batch_crop_scf_padding_ub[11])

    def get_batch_switch(self):
        """
        get crop, scf, padding switch
        """
        head_offset_crop_switch = 0
        with self.tik_instance.new_stmt_scope():
            batch_switch_ub = self.batch_params_ub[
                              head_offset_crop_switch:Constant.HEAD_OFFSET_CROP_START_W].reinterpret_cast_to("int8")

            self.crop_switch.set_as(batch_switch_ub[0])
            self.scf_switch.set_as(batch_switch_ub[1])
            self.padding_switch.set_as(batch_switch_ub[2])

    def get_batch_params(self, batch_id):
        """
        get batch params
        """
        # move batch params
        self.tik_instance.data_move(self.batch_params_ub,
                                    self.dync_param_gm[Constant.PARAM_HEAD_STRUCT_SIZE + \
                                    Constant.PARAM_BATCH_STRUCT_SIZE * batch_id], 0, 1, \
                                    Constant.PARAM_BATCH_STRUCT_SIZE // Constant.BLOCK_BYTES, 0, 0)

        self.get_batch_switch()
        self.get_crop_scf_padding_params()
        self.get_dtc_params()

    def set_csc_matrix_and_bias_hisi(self):
        """
        set csc matrix, out bias and in bias for ihisi
        """
        matrix_r0_c0_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r0_c0_cs", init_value=0)
        matrix_r0_c1_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r0_c1_cs", init_value=0)
        matrix_r0_c2_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r0_c2_cs", init_value=0)
        matrix_r1_c0_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r1_c0_cs", init_value=0)
        matrix_r1_c1_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r1_c1_cs", init_value=0)
        matrix_r1_c2_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r1_c2_cs", init_value=0)
        matrix_r2_c0_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r2_c0_cs", init_value=0)
        matrix_r2_c1_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r2_c1_cs", init_value=0)
        matrix_r2_c2_cs = self.tik_instance.Scalar(dtype="int16", name="matrix_r2_c2_cs", init_value=0)

        output_bias_r0_cs = self.tik_instance.Scalar(dtype="uint8", name="output_bias_r0_cs", init_value=0)
        output_bias_r1_cs = self.tik_instance.Scalar(dtype="uint8", name="output_bias_r1_cs", init_value=0)
        output_bias_r2_cs = self.tik_instance.Scalar(dtype="uint8", name="output_bias_r2_cs", init_value=0)

        input_bias_r0_cs = self.tik_instance.Scalar(dtype="uint8", name="input_bias_r0_cs", init_value=0)
        input_bias_r1_cs = self.tik_instance.Scalar(dtype="uint8", name="input_bias_r1_cs", init_value=0)
        input_bias_r2_cs = self.tik_instance.Scalar(dtype="uint8", name="input_bias_r2_cs", init_value=0)

        # YUV420SP_U8, YUYV_U8, YUV422SP_U8, AYUV444_U8, YUV400
        with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_YUV420SP,
                                                self.input_format == Constant.FORMAT_YUYV,
                                                self.input_format == Constant.FORMAT_YUV422SP,
                                                self.input_format == Constant.FORMAT_YUV400,
                                                self.input_format == Constant.FORMAT_AYUV444)):
            with self.tik_instance.if_scope(self.csc_switch > 0):
                matrix_r0_c0_cs.set_as(self.matrix_r2_c0 * 4)
                matrix_r0_c1_cs.set_as(self.matrix_r2_c1 * 4)
                matrix_r0_c2_cs.set_as(self.matrix_r2_c2 * 4)
                matrix_r1_c0_cs.set_as(self.matrix_r1_c0 * 4)
                matrix_r1_c1_cs.set_as(self.matrix_r1_c1 * 4)
                matrix_r1_c2_cs.set_as(self.matrix_r1_c2 * 4)
                matrix_r2_c0_cs.set_as(self.matrix_r0_c0 * 4)
                matrix_r2_c1_cs.set_as(self.matrix_r0_c1 * 4)
                matrix_r2_c2_cs.set_as(self.matrix_r0_c2 * 4)

                input_bias_r0_cs.set_as(self.input_bias_r2)
                input_bias_r1_cs.set_as(self.input_bias_r1)
                input_bias_r2_cs.set_as(self.input_bias_r0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.input_format != Constant.FORMAT_YUV400):
                    self.csc_switch.set_as(1)

                # spr_2->bits.csc_matrix_r0_c2, 1 << 10 & 0xffff
                matrix_r0_c2_cs.set_as(1024)

                # spr_3->bits.csc_matrix_r1_c1
                matrix_r1_c1_cs.set_as(1024)
                # spr_3->bits.csc_matrix_r2_c0
                matrix_r2_c0_cs.set_as(1024)

        # XRGB8888_U8, RGB888_U8, ARGB8888_U8
        with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_XRGB8888,
                                                self.input_format == Constant.FORMAT_RGB888,
                                                self.input_format == Constant.FORMAT_ARGB8888)):
            with self.tik_instance.if_scope(self.csc_switch > 0):
                matrix_r0_c0_cs.set_as(self.matrix_r2_c2 * 4)
                matrix_r0_c1_cs.set_as(self.matrix_r2_c1 * 4)
                matrix_r0_c2_cs.set_as(self.matrix_r2_c0 * 4)
                matrix_r1_c0_cs.set_as(self.matrix_r1_c2 * 4)
                matrix_r1_c1_cs.set_as(self.matrix_r1_c1 * 4)
                matrix_r1_c2_cs.set_as(self.matrix_r1_c0 * 4)
                matrix_r2_c0_cs.set_as(self.matrix_r0_c2 * 4)
                matrix_r2_c1_cs.set_as(self.matrix_r0_c1 * 4)
                matrix_r2_c2_cs.set_as(self.matrix_r0_c0 * 4)

                output_bias_r0_cs.set_as(self.output_bias_r0)
                output_bias_r1_cs.set_as(self.output_bias_r1)
                output_bias_r2_cs.set_as(self.output_bias_r2)

        self.csc_matrix = [[matrix_r0_c0_cs, matrix_r0_c1_cs, matrix_r0_c2_cs],
                           [matrix_r1_c0_cs, matrix_r1_c1_cs, matrix_r1_c2_cs],
                           [matrix_r2_c0_cs, matrix_r2_c1_cs, matrix_r2_c2_cs]]
        self.csc_in_bias = [input_bias_r0_cs, input_bias_r1_cs, input_bias_r2_cs]
        self.csc_out_bias = [output_bias_r0_cs, output_bias_r1_cs, output_bias_r2_cs]

    def set_csc_matrix_and_bias(self):
        """
        set csc matrix, out bias and in bias
        """
        self.csc_matrix = [[self.matrix_r0_c0, self.matrix_r0_c1, self.matrix_r0_c2],
                           [self.matrix_r1_c0, self.matrix_r1_c1, self.matrix_r1_c2],
                           [self.matrix_r2_c0, self.matrix_r2_c1, self.matrix_r2_c2]]
        self.csc_in_bias = [self.input_bias_r0, self.input_bias_r1, self.input_bias_r2]
        self.csc_out_bias = [self.output_bias_r0, self.output_bias_r1, self.output_bias_r2]

    def set_swap_list(self):
        """
        set swap list
        """
        rb_swap = self.tik_instance.Scalar(dtype="uint8", name="rb_swap", init_value=0)
        uv_swap = self.tik_instance.Scalar(dtype="uint8", name="uv_swap", init_value=0)
        ax_swap = self.tik_instance.Scalar(dtype="uint8", name="ax_swap", init_value=0)

        with self.tik_instance.if_scope(self.rbuv_swap_switch > 0):
            with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_YUV420SP,
                                                    self.input_format == Constant.FORMAT_YUV422SP,
                                                    self.input_format == Constant.FORMAT_AYUV444,
                                                    self.input_format == Constant.FORMAT_YUYV)):
                uv_swap.set_as(1)
            with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_XRGB8888,
                                                    self.input_format == Constant.FORMAT_RGB888,
                                                    self.input_format == Constant.FORMAT_ARGB8888)):
                rb_swap.set_as(1)

        with self.tik_instance.if_scope(self.ax_swap_switch > 0):
            with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_XRGB8888,
                                                    self.input_format == Constant.FORMAT_ARGB8888,
                                                    self.input_format == Constant.FORMAT_AYUV444)):
                ax_swap.set_as(1)

        self.swap_list = [rb_swap, uv_swap, ax_swap]

    def set_dtc_list_hisi(self):
        """
        set dtc list, ihisi
        """
        dtc_mean_chn0_tmp = self.tik_instance.Scalar(dtype="int16", name="dtc_mean_chn0_tmp")
        dtc_mean_chn2_tmp = self.tik_instance.Scalar(dtype="int16", name="dtc_mean_chn2_tmp")
        dtc_min_chn0_tmp = self.tik_instance.Scalar(dtype="float16", name="dtc_min_chn0_tmp")
        dtc_min_chn2_tmp = self.tik_instance.Scalar(dtype="float16", name="dtc_min_chn2_tmp")
        dtc_reci_chn0_tmp = self.tik_instance.Scalar(dtype="float16", name="dtc_reci_chn0_tmp")
        dtc_reci_chn2_tmp = self.tik_instance.Scalar(dtype="float16", name="dtc_reci_chn2_tmp")

        with self.tik_instance.if_scope(self.input_format == Constant.FORMAT_YUV400):
            dtc_mean_chn0_tmp.set_as(self.dtc_mean_chn2)
            dtc_mean_chn2_tmp.set_as(self.dtc_mean_chn0)
            dtc_min_chn0_tmp.set_as(self.dtc_min_chn2)
            dtc_min_chn2_tmp.set_as(self.dtc_min_chn0)
            dtc_reci_chn0_tmp.set_as(self.dtc_reci_chn2)
            dtc_reci_chn2_tmp.set_as(self.dtc_reci_chn0)
        with self.tik_instance.else_scope():
            dtc_mean_chn0_tmp.set_as(self.dtc_mean_chn0)
            dtc_mean_chn2_tmp.set_as(self.dtc_mean_chn2)
            dtc_min_chn0_tmp.set_as(self.dtc_min_chn0)
            dtc_min_chn2_tmp.set_as(self.dtc_min_chn2)
            dtc_reci_chn0_tmp.set_as(self.dtc_reci_chn0)
            dtc_reci_chn2_tmp.set_as(self.dtc_reci_chn2)

        self.dtc_mean_list = [dtc_mean_chn0_tmp, self.dtc_mean_chn1, dtc_mean_chn2_tmp, self.dtc_mean_chn3]
        self.dtc_min_list = [dtc_min_chn0_tmp, self.dtc_min_chn1, dtc_min_chn2_tmp, self.dtc_min_chn3]
        self.dtc_reci_list = [dtc_reci_chn0_tmp, self.dtc_reci_chn1, dtc_reci_chn2_tmp, self.dtc_reci_chn3]

    def set_dtc_list(self):
        """
        set dtc list
        """
        self.dtc_mean_list = [self.dtc_mean_chn0, self.dtc_mean_chn1, self.dtc_mean_chn2, self.dtc_mean_chn3]
        self.dtc_min_list = [self.dtc_min_chn0, self.dtc_min_chn1, self.dtc_min_chn2, self.dtc_min_chn3]
        self.dtc_reci_list = [self.dtc_reci_chn0, self.dtc_reci_chn1, self.dtc_reci_chn2, self.dtc_reci_chn3]

    def set_src0_offset(self, batch_id, src0_offset):
        """
        calculate src0 offset
        """
        with self.tik_instance.if_scope(self.input_format == Constant.FORMAT_YUV420SP):
            src0_offset.set_as(batch_id * ((self.src_image_size_h * self.src_image_size_w * 3) // 2))

        with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_YUV422SP,
                                                self.input_format == Constant.FORMAT_YUYV)):
            src0_offset.set_as(batch_id * (self.src_image_size_h * self.src_image_size_w * 2))

        with self.tik_instance.if_scope(tik.any(self.input_format == Constant.FORMAT_XRGB8888,
                                                self.input_format == Constant.FORMAT_ARGB8888,
                                                self.input_format == Constant.FORMAT_AYUV444)):
            src0_offset.set_as(batch_id * (self.src_image_size_h * self.src_image_size_w * 4))

        with self.tik_instance.if_scope(self.input_format == Constant.FORMAT_RGB888):
            src0_offset.set_as(batch_id * (self.src_image_size_h * self.src_image_size_w * 3))

        with self.tik_instance.if_scope(self.input_format == Constant.FORMAT_YUV400):
            src0_offset.set_as(batch_id * (self.src_image_size_h * self.src_image_size_w))

    def vector_conv(self, dst_ub, src_ub, elems):
        """
        convert float16 to uint8 or int8

        Parameters
        ----------
        dst_ub: destination ub
        src_ub: source ub
        elems: number of elements

        Returns
        -------
        None
        """
        one_cnt = 128
        max_repeat = 255
        repeats = elems // one_cnt
        elems_tail = elems % one_cnt
        repeats_loop = repeats // max_repeat
        repeats_tail = repeats % max_repeat

        with self.tik_instance.if_scope(repeats_loop > 0):
            with self.tik_instance.for_range(0, repeats_loop) as loop_i:
                repeat_offset = one_cnt * max_repeat * loop_i
                self.tik_instance.vconv(one_cnt, '', dst_ub[repeat_offset], src_ub[repeat_offset],
                                        max_repeat, 1, 1, 4, 8)

        repeats_tail_offset = one_cnt * max_repeat * repeats_loop
        with self.tik_instance.if_scope(repeats_tail > 0):
            self.tik_instance.vconv(one_cnt, '', dst_ub[repeats_tail_offset], src_ub[repeats_tail_offset],
                                    repeats_tail, 1, 1, 4, 8)

        elems_tail_offset = repeats_tail_offset + one_cnt * repeats_tail
        with self.tik_instance.if_scope(elems_tail > 0):
            self.tik_instance.vconv(elems_tail, '', dst_ub[elems_tail_offset], src_ub[elems_tail_offset],
                                    1, 1, 1, 4, 8)

    def vector_dup_fp16(self, data_ub, elems, padding_value):
        """
        set all data to padding_value, float16

        Parameters
        ----------
        data_ub: destination ub
        elems: number of elements
        padding_value: padding value

        Returns
        -------
        None
        """
        one_cnt = 128
        max_repeat = 255
        repeats = elems // one_cnt
        elems_tail = elems % one_cnt
        repeats_loop = repeats // max_repeat
        repeats_tail = repeats % max_repeat

        with self.tik_instance.if_scope(repeats_loop > 0):
            with self.tik_instance.for_range(0, repeats_loop) as loop_i:
                repeat_offset = one_cnt * max_repeat * loop_i
                self.tik_instance.vector_dup(one_cnt, data_ub[repeat_offset], padding_value, max_repeat, 1, 8)

        repeats_tail_offset = one_cnt * max_repeat * repeats_loop
        with self.tik_instance.if_scope(repeats_tail > 0):
            self.tik_instance.vector_dup(one_cnt, data_ub[repeats_tail_offset], padding_value, repeats_tail, 1, 8)

        elems_tail_offset = repeats_tail_offset + one_cnt * repeats_tail
        with self.tik_instance.if_scope(elems_tail > 0):
            self.tik_instance.vector_dup(elems_tail, data_ub[elems_tail_offset], padding_value, 1, 1, 8)

    def process_padding_s8(self, padding_size, gm_offset):
        """
        top or bottom padding process, uint8 or int8
        """
        with self.tik_instance.new_stmt_scope():
            padding_ub = self.tik_instance.Tensor(self.output_dtype, (self.max_s8_padding_elems,), name="padding_ub",
                                                  scope=tik.scope_ubuf)
            temp_ub = self.tik_instance.Tensor("float16", (self.max_s8_padding_elems,), name="temp_ub",
                                               scope=tik.scope_ubuf)
            padding_elems = self.output_c1 * padding_size * self.output_w * self.c0

            with self.tik_instance.if_scope(padding_elems <= self.max_s8_padding_elems):
                self.vector_dup_fp16(temp_ub, padding_elems, 0)
                self.vector_conv(padding_ub, temp_ub, padding_elems)
                self.tik_instance.data_move(self.output_gm[gm_offset], padding_ub, 0, 1,
                                            ceil_value(padding_elems, self.block_elems), 0, 0)

            with self.tik_instance.else_scope():
                tiling_h = (self.max_s8_padding_elems // self.c0) // self.output_w
                h_loop = padding_size // tiling_h
                tail_h = padding_size % tiling_h

                one_loop_elems = self.output_c1 * tiling_h * self.output_w * self.c0
                with self.tik_instance.for_range(0, h_loop) as loop_i:
                    loop_i_offset = gm_offset + loop_i * one_loop_elems
                    self.vector_dup_fp16(temp_ub, one_loop_elems, 0)
                    self.vector_conv(padding_ub, temp_ub, one_loop_elems)
                    self.tik_instance.data_move(self.output_gm[loop_i_offset], padding_ub, 0, 1,
                                                ceil_value(one_loop_elems, self.block_elems), 0, 0)
                with self.tik_instance.if_scope(tail_h > 0):
                    tail_elems = self.output_c1 * tail_h * self.output_w * self.c0
                    tail_offset = gm_offset + h_loop * one_loop_elems
                    self.vector_dup_fp16(temp_ub, tail_elems, 0)
                    self.vector_conv(padding_ub, temp_ub, tail_elems)
                    self.tik_instance.data_move(self.output_gm[tail_offset], padding_ub, 0, 1,
                                                ceil_value(tail_elems, self.block_elems), 0, 0)

    def process_padding_fp16(self, padding_size, gm_offset):
        """
        top or bottom padding process, float16
        """
        with self.tik_instance.new_stmt_scope():
            padding_ub = self.tik_instance.Tensor(self.output_dtype, (self.ub_max_elems,), name="padding_ub",
                                                  scope=tik.scope_ubuf)
            padding_elems = self.output_c1 * padding_size * self.output_w * self.c0

            with self.tik_instance.if_scope(padding_elems <= self.ub_max_elems):
                self.vector_dup_fp16(padding_ub, padding_elems, 0)
                self.tik_instance.data_move(self.output_gm[gm_offset], padding_ub, 0, 1,
                                            ceil_value(padding_elems, self.block_elems), 0, 0)

            with self.tik_instance.else_scope():
                tiling_h = (self.ub_max_elems // self.c0) // self.output_w
                h_loop = padding_size // tiling_h
                tail_h = padding_size % tiling_h

                one_loop_elems = self.output_c1 * tiling_h * self.output_w * self.c0
                with self.tik_instance.for_range(0, h_loop) as loop_i:
                    loop_i_offset = gm_offset + loop_i * one_loop_elems
                    self.vector_dup_fp16(padding_ub, one_loop_elems, 0)
                    self.tik_instance.data_move(self.output_gm[loop_i_offset], padding_ub, 0, 1,
                                                ceil_value(one_loop_elems, self.block_elems), 0, 0)
                with self.tik_instance.if_scope(tail_h > 0):
                    tail_elems = self.output_c1 * tail_h * self.output_w * self.c0
                    tail_offset = gm_offset + h_loop * one_loop_elems
                    self.vector_dup_fp16(padding_ub, tail_elems, 0)
                    self.tik_instance.data_move(self.output_gm[tail_offset], padding_ub, 0, 1,
                                                ceil_value(tail_elems, self.block_elems), 0, 0)

    def process_padding(self, padding_size, gm_offset):
        """
        top or bottom padding process

        Parameters
        ----------
        padding_size: top or bottom padding size
        gm_offset: offset of gm

        Returns
        -------
        None
        """
        if self.output_dtype == "float16":
            self.process_padding_fp16(padding_size, gm_offset)
        else:
            self.process_padding_s8(padding_size, gm_offset)

    def move_data_from_l1_to_gm(self, elems_num, data_l1, data_ub, gm_output_offset):
        """
        move data from L1 to gm

        Parameters
        ----------
        elems_num: number of elements
        data_l1: data L1
        data_ub: data UB
        gm_output_offset: offset of gm

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(elems_num <= self.ub_max_elems):
            burst_len = ceil_value(elems_num, self.block_elems)
            self.tik_instance.data_move(data_ub, data_l1, 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.output_gm[gm_output_offset], data_ub, 0, 1, burst_len, 0, 0)

        with self.tik_instance.else_scope():
            loops = elems_num // self.ub_max_elems
            tail_num = elems_num % self.ub_max_elems
            one_burst_len = self.ub_max_elems // self.block_elems

            with self.tik_instance.for_range(0, loops) as loop_i:
                l1_offset = loop_i * self.ub_max_elems
                self.tik_instance.data_move(data_ub, data_l1[l1_offset], 0, 1, one_burst_len, 0, 0)
                self.tik_instance.data_move(self.output_gm[gm_output_offset + l1_offset], data_ub,
                                            0, 1, one_burst_len, 0, 0)

            with self.tik_instance.if_scope(tail_num > 0):
                burst_len_tail = ceil_value(tail_num, self.block_elems)
                l1_offset = loops * self.ub_max_elems
                self.tik_instance.data_move(data_ub, data_l1[l1_offset], 0, 1, burst_len_tail, 0, 0)
                self.tik_instance.data_move(self.output_gm[gm_output_offset + l1_offset], data_ub,
                                            0, 1, burst_len_tail, 0, 0)

    def set_src_info(self):
        """
        set src_info
        """
        src_info = {
            'src_horizontal_size': self.src_image_size_w,
            'src_vertical_size': self.src_image_size_h
        }

        return src_info

    def set_fix_params(self):
        """
        set csc_info, dtc_info, channel_pad_info
        """
        csc_info = {
            'format_convert': 0,
            'csc_matrix': self.csc_matrix,
            'csc_out_bias': self.csc_out_bias,
            'csc_in_bias': self.csc_in_bias
        }
        dtc_info = {
            'dtc_mean_type': 0,
            'dtc_mean': self.dtc_mean_list,
            'dtc_min': self.dtc_min_list,
            'dtc_var': self.dtc_reci_list,
            'raw_to_f16_n': 0
        }
        channel_pad_info = {
            'channel_pad_mode': self.channel_pad_mode,
            'channel_pad_value': self.c_padding_value_zero
        }

        return csc_info, dtc_info, channel_pad_info

    def split_h_internal_impl(self, batch_id, src0_offset, split_h, function_switch, params):
        """
        split height internal impl.
        params: (h_loop_i, tiling_h, last_tiling_h, tail_h)
        """
        src_info = self.set_src_info()
        csc_info, dtc_info, channel_pad_info = self.set_fix_params()
        pre_clip_info = None
        post_clip_info = None
        flip_mode = 0
        stretch_info = None
        raw_info = None
        sid = 0

        h_loop_i, tiling_h, last_tiling_h, tail_h = params

        with self.tik_instance.new_stmt_scope():
            data_l1 = self.tik_instance.Tensor(self.output_dtype, (self.l1_max_elems,), name="data_l1",
                                               scope=tik.scope_cbuf)
            data_ub = self.tik_instance.Tensor(self.output_dtype, (self.ub_max_elems,), name="data_ub",
                                               scope=tik.scope_ubuf)

            tiling_crop_size_h = self.tik_instance.Scalar(dtype="int32", name="tiling_crop_size_h")
            tiling_crop_size_h.set_as(split_h)
            tiling_crop_start_pos_h = self.tik_instance.Scalar(dtype="int32", name="tiling_crop_start_pos_h")
            with self.tik_instance.if_scope(tail_h == 0):
                tiling_crop_start_pos_h.set_as(self.crop_start_pos_h + h_loop_i * tiling_h)
            with self.tik_instance.else_scope():
                tiling_crop_start_pos_h.set_as(self.crop_start_pos_h + h_loop_i * tiling_h + last_tiling_h)

            crop_info = {
                'dst_horizontal_size': self.crop_size_w,
                'dst_vertical_size': tiling_crop_size_h,
                'crop_horizontal_start': self.crop_start_pos_w,
                'crop_vertical_start': tiling_crop_start_pos_h,
                'single_line_enable': 0
            }
            scf_info = {
                'scf_horizontal_size': self.scf_output_size_w,
                'scf_vertical_size': split_h,
                'scaling_mode': 0,
                'scf_horizontal_start': 0,
                'scf_vertical_start': 0
            }
            area_pad_info = {
                'area_pad_mode': 0,
                'top_pad_rows': 0,
                'botton_pad_rows': 0,
                'left_pad_cols': self.padding_size_left,
                'right_pad_cols': self.padding_size_right,
                'channel0_pad_value': self.area_padding_value_zero,
                'channel1_pad_value': self.area_padding_value_zero,
                'channel2_pad_value': self.area_padding_value_zero,
                'channel3_pad_value': self.area_padding_value_zero
            }

            self.tik_instance.load_image(data_l1, self.data_gm[src0_offset], None, self.input_format - 1,
                                         function_switch, src_info, crop_info, pre_clip_info, self.swap_list,
                                         csc_info, scf_info, post_clip_info, dtc_info, flip_mode,
                                         channel_pad_info, area_pad_info, stretch_info, raw_info, sid)

            # move data to gm from L1
            elems_num = self.output_c1 * split_h * self.output_w * self.c0
            one_loop_elems = self.output_c1 * tiling_h * self.output_w * self.c0
            last_tiling_h_elems = self.output_c1 * last_tiling_h * self.output_w * self.c0
            one_batch_offset = self.output_c1 * self.output_h * self.output_w * self.c0
            gm_output_offset = self.tik_instance.Scalar(dtype="int32", name="gm_output_offset")
            with self.tik_instance.if_scope(tail_h == 0):
                gm_output_offset.set_as(batch_id * one_batch_offset + self.padding_size_top * self.output_w * self.c0
                                        + one_loop_elems * h_loop_i)
            with self.tik_instance.else_scope():
                gm_output_offset.set_as(batch_id * one_batch_offset + self.padding_size_top * self.output_w * self.c0
                                        + one_loop_elems * h_loop_i + last_tiling_h_elems)

            self.move_data_from_l1_to_gm(elems_num, data_l1, data_ub, gm_output_offset)

    def split_h_impl(self, batch_id, src0_offset, actual_hw_size, function_switch):
        """
        split height impl

        Parameters
        ----------
        batch_id: batch id
        src0_offset: src0 offset
        actual_hw_size: actual hw size
        function_switch: function_switch

        Returns
        -------
        None
        """
        tiling_h = self.tik_instance.Scalar(dtype="int32", name="tiling_h")
        last_tiling_h = self.tik_instance.Scalar(dtype="int32", name="last_tiling_h", init_value=0)
        h_loop = self.tik_instance.Scalar(dtype="int32", name="h_loop")
        tail_h = self.tik_instance.Scalar(dtype="int32", name="tail_h")
        tiling_h.set_as(self.max_hw_l1_size // self.output_w)
        with self.tik_instance.if_scope(tiling_h % 2 == 1):
            tiling_h.set_as(self.max_hw_l1_size // self.output_w - 1)
        h_loop.set_as(actual_hw_size // self.output_w // tiling_h)
        tail_h.set_as((actual_hw_size // self.output_w) % tiling_h)

        with self.tik_instance.if_scope(tik.all(h_loop > 0, tail_h > 0, tail_h < 8)):
            h_loop.set_as(h_loop - 1)
            with self.tik_instance.if_scope(tail_h % 2 == 1):
                last_tiling_h.set_as(tiling_h - 9 + tail_h)
                tail_h.set_as(9)
            with self.tik_instance.else_scope():
                last_tiling_h.set_as(tiling_h - 8 + tail_h)
                tail_h.set_as(8)

        with self.tik_instance.for_range(0, h_loop) as h_i:
            self.split_h_internal_impl(batch_id, src0_offset, tiling_h, function_switch,
                                       (h_i, tiling_h, last_tiling_h, 0))

        with self.tik_instance.if_scope(last_tiling_h > 0):
            self.split_h_internal_impl(batch_id, src0_offset, last_tiling_h, function_switch,
                                       (h_loop, tiling_h, last_tiling_h, 0))

        # tail h
        with self.tik_instance.if_scope(tail_h > 0):
            self.split_h_internal_impl(batch_id, src0_offset, tail_h, function_switch,
                                       (h_loop, tiling_h, last_tiling_h, tail_h))

    def no_split_impl(self, batch_id, src0_offset, actual_hw_size, function_switch):
        """
        height does not need to be split

        Parameters
        ----------
        batch_id: batch id
        src0_offset: src0 offset
        actual_hw_size: actual hw size
        function_switch: function_switch

        Returns
        -------
        None
        """
        src_info = self.set_src_info()
        csc_info, dtc_info, channel_pad_info = self.set_fix_params()
        pre_clip_info = None
        post_clip_info = None
        flip_mode = 0
        stretch_info = None
        raw_info = None
        sid = 0

        with self.tik_instance.new_stmt_scope():
            data_l1 = self.tik_instance.Tensor(self.output_dtype, (self.l1_max_elems,), name="data_l1",
                                               scope=tik.scope_cbuf)
            data_ub = self.tik_instance.Tensor(self.output_dtype, (self.ub_max_elems,), name="data_ub",
                                               scope=tik.scope_ubuf)
            crop_info = {
                'dst_horizontal_size': self.crop_size_w,
                'dst_vertical_size': self.crop_size_h,
                'crop_horizontal_start': self.crop_start_pos_w,
                'crop_vertical_start': self.crop_start_pos_h,
                'single_line_enable': 0
            }
            scf_info = {
                'scf_horizontal_size': self.scf_output_size_w,
                'scf_vertical_size': self.scf_output_size_h,
                'scaling_mode': 0,
                'scf_horizontal_start': 0,
                'scf_vertical_start': 0
            }
            area_pad_info = {
                'area_pad_mode': 0,
                'top_pad_rows': 0,
                'botton_pad_rows': 0,
                'left_pad_cols': self.padding_size_left,
                'right_pad_cols': self.padding_size_right,
                'channel0_pad_value': self.area_padding_value_zero,
                'channel1_pad_value': self.area_padding_value_zero,
                'channel2_pad_value': self.area_padding_value_zero,
                'channel3_pad_value': self.area_padding_value_zero
            }

            self.tik_instance.load_image(data_l1, self.data_gm[src0_offset], None, self.input_format - 1,
                                         function_switch, src_info, crop_info, pre_clip_info, self.swap_list,
                                         csc_info, scf_info, post_clip_info, dtc_info, flip_mode,
                                         channel_pad_info, area_pad_info, stretch_info, raw_info, sid)

            # move data to gm from L1
            elems_num = self.output_c1 * actual_hw_size * self.c0
            one_batch_offset = self.output_c1 * self.output_h * self.output_w * self.c0
            gm_output_offset = batch_id * one_batch_offset + self.padding_size_top * self.output_w * self.c0
            self.move_data_from_l1_to_gm(elems_num, data_l1, data_ub, gm_output_offset)

    def process_one_batch(self, batch_id, src0_offset):
        """
        process one batch

        Parameters
        ----------
        batch_id: batch id
        src0_offset: src0 offset

        Returns
        -------
        None
        """
        tik_inst = self.tik_instance
        self.get_batch_params(batch_id)

        if self.ihisi:
            self.set_dtc_list_hisi()
        else:
            self.set_dtc_list()

        # function_switch: 1-crop, 4-swap, 8-csc, 16-resize, 64-dtc, 256-area padding, 512-c padding
        # c padding: 512
        function_switch = tik_inst.Scalar(dtype="uint32", name="function_switch", init_value=512)
        # the value of dtc is 64
        if self.output_dtype != "uint8":
            function_switch.set_as(function_switch + 64)

        actual_hw_size = tik_inst.Scalar(dtype="int32", name="actual_hw_size")
        actual_hw_size.set_as(self.src_image_size_h * self.src_image_size_w)
        output_h_no_pad = tik_inst.Scalar(dtype="int32", name="output_h_no_pad")

        # the value of crop is 1
        function_switch.set_as(function_switch + 1)
        with tik_inst.if_scope(self.crop_switch > 0):
            actual_hw_size.set_as(self.crop_size_h * self.crop_size_w)

        with tik_inst.if_scope(tik.any(self.rbuv_swap_switch > 0, self.ax_swap_switch > 0)):
            # the value of swap is 4
            function_switch.set_as(function_switch + 4)

        with tik_inst.if_scope(self.csc_switch > 0):
            # the value of csc is 8
            function_switch.set_as(function_switch + 8)

        with tik_inst.if_scope(self.padding_switch > 0):
            # area padding: 256
            function_switch.set_as(function_switch + 256)
            output_h_no_pad.set_as(self.output_h - self.padding_size_top - self.padding_size_bottom)
            actual_hw_size.set_as(output_h_no_pad * self.output_w)

            # top padding process
            with tik_inst.if_scope(self.padding_size_top > 0):
                top_offset = batch_id * (self.output_c1 * self.output_h * self.output_w * self.c0)
                self.process_padding(self.padding_size_top, top_offset)

        with tik_inst.if_scope(actual_hw_size < self.max_hw_l1_size):
            self.no_split_impl(batch_id, src0_offset, actual_hw_size, function_switch)
        with tik_inst.else_scope():
            # height needs to be split
            self.split_h_impl(batch_id, src0_offset, actual_hw_size, function_switch)

        with tik_inst.if_scope(self.padding_switch > 0):
            # bottom padding process
            with tik_inst.if_scope(self.padding_size_bottom > 0):
                bottom_offset = batch_id * (self.output_c1 * self.output_h * self.output_w * self.c0) + \
                                (self.output_h - self.padding_size_bottom) * self.output_w * self.c0
                self.process_padding(self.padding_size_bottom, bottom_offset)

    def process_one_core(self, batch_offset, batch):
        """
        one core process

        Parameters
        ----------
        batch_offset: batch offset
        batch: number of batch processed

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, batch) as batch_i:
            batch_id = batch_offset + batch_i
            # cal src0 offset
            src0_offset = self.tik_instance.Scalar(dtype="int32", name="src0_offset")
            self.set_src0_offset(batch_id, src0_offset)

            self.process_one_batch(batch_id, src0_offset)

    def aipp_dynamic_shape_compute(self):
        """
        main process of aipp dynamic shape

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tik_inst = self.tik_instance
        with tik_inst.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            self.batch_params_ub = tik_inst.Tensor(self.param_dtype, (Constant.PARAM_BATCH_STRUCT_SIZE,),
                                                   name="batch_params_ub", scope=tik.scope_ubuf)
            self.init_batch_params()

            # get tiling data
            self.tiling_ub = tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            tik_inst.data_move(self.tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // 4, 0, 0)
            self.get_tiling_args()

            with tik_inst.if_scope(block_id < self.need_core_num):
                self.get_public_params()

                if self.ihisi:
                    self.set_csc_matrix_and_bias_hisi()
                else:
                    self.set_csc_matrix_and_bias()

                self.set_swap_list()

                with tik_inst.if_scope(block_id < self.need_core_num - 1):
                    self.process_one_core(block_id * self.batch_each_core, self.batch_each_core)

                # last core
                with tik_inst.else_scope():
                    self.process_one_core(block_id * self.batch_each_core, self.batch_last_core)


@register_operator("Aipp")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def aipp(input_data, input_dync_param, output_data, aipp_config_json="./aipp.cfg", kernel_name="aipp"):
    """
    Operation for aipp.

    Parameters
    ----------
    input_data: dict of input, include shape and dtype, dtype support uint8
    input_dync_param: dict of dynamic parameter, include shape and dtype, dtype support uint8
    aipp_config_json: json of aipp config
    kernel_name: cce kernel name, default value is aipp

    Returns
    -------
    tik_instance
    """
    check_input_params(input_data, input_dync_param, output_data)

    obj = Aipp(input_data, input_dync_param, output_data)
    obj.aipp_dynamic_shape_compute()

    tik_inst = obj.tik_instance
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=(obj.data_gm, obj.dync_param_gm),
                      outputs=(obj.output_gm,),
                      flowtable=(obj.tiling_gm,), enable_l2=True)

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {"core_num": obj.core_num})

    return tik_inst
