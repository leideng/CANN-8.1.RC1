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
platform adapter
"""

from tbe.dsl.base import operation as tbe_operation
import tbe as platform_tbe
import tbe.common.buildcfg as tbe_build
import tbe.common.register as tbe_register
from tbe import tik as tbe_tik
from tbe import tvm as tbe_tvm
from tbe.common import platform
from tbe.common import buildcfg
from tbe.common.platform import platform_info
from tbe.common.utils import log as tbe_log
from te.platform.fusion_manager import fusion_manager as tbe_fusion_manager
from te.lang.cce import tuple_sum as te_tuple_sum
from tbe.dsl.instrinsic import cce_util

register_operator = tbe_register.register_operator


def register_operator_compute(op_type, op_mode="dynamic", support_fusion=False, **kwargs):
    """
    register op compute func

    Parameters
    ----------
    op_type : string
        op_func_name(old process) or op type(new process)
    op_mode: string
        dynamic or static shape
    support_fusion: bool
        support dynamic shape UB fusion

    Returns
    -------
    decorator : decorator
        decorator to register compute func
    """
    return tbe_register.register_op_compute(op_type, op_mode, support_fusion, **kwargs)


class OpPatternMode:
    """
    op pattern mode
    """
    NONE = ""
    ELEWISE = "elewise"
    ELEWISE_WITH_BROADCAST = "broadcast"
    REDUCE = "reduce"


class OpImplMode:
    """
    op implement mode high_performance or high_precision
    """
    HIGH_PERFORMANCE = "high_performance"
    HIGH_PRECISION = "high_precision"


class TbeContextKey:
    """
    TbeContextKey
    """
    PATTERN = "pattern"


class PlatformApi:
    """
    platform API
    """
    # Instructions Strings
    EXP = "vector_exp"
    RELU = "vector_relu"
    REC = "vector_rec"
    LN = "vector_ln"
    ABS = "vector_abs"
    SQRT = "vector_sqrt"
    RSQRT = "vector_rsqrt"
    NOT = "vector_not"
    DUP = "vector_dup"
    MUL = "vector_mul"
    ADD = "vector_add"
    SUB = "vector_sub"
    DIV = "vector_div"
    MAX = "vector_max"
    MIN = "vector_min"
    MULVS = "vector_muls"
    ADDVS = "vector_adds"
    MAXVS = "vector_maxs"
    MINVS = "vector_mins"
    LRELU = "vector_lrelu"
    EQ = "vector_eq"
    NE = "vector_ne"
    GE = "vector_ge"
    LE = "vector_le"
    GT = "vector_gt"
    LT = "vector_lt"
    EQVS = "vector_eqs"
    NEVS = "vector_nes"
    GEVS = "vector_ges"
    LEVS = "vector_les"
    GTVS = "vector_gts"
    LTVS = "vector_lts"
    AND = "vector_and"
    OR = "vector_or"
    MULCONV = "vector_mul_conv"
    ADDRELU = "vector_addrelu"
    SUBRELU = "vector_subrelu"
    ADDRELUCONV = "vector_addrelu_conv"
    SUBRELUCONV = "vector_subrelu_conv"
    SHR = "vector_shr"
    SHR_ROUND = "vector_shr_round"
    SHL = "vector_shl"
    MADD = "vector_madd"
    MADDRELU = "vector_maddrelu"
    MLA = "vector_mla"
    AXPY = "vector_axpy"
    CAST = "vector_conv"
    CAST_VDEQ = "vector_conv_vdeq"
    CAST_RINT = "vector_conv_rint"
    CAST_ROUND = "vector_conv_round"
    CAST_FLOOR = "vector_conv_floor"
    CAST_CEIL = "vector_conv_ceil"
    CAST_TRUNC = "vector_conv_trunc"
    CAST_ROUNDING = "vector_conv_rounding"
    TCAST = "vector_tconv"
    TCAST_RINT = "vector_tconv_rint"
    TCAST_ROUND = "vector_tconv_round"
    TCAST_FLOOR = "vector_tconv_floor"
    TCAST_CEIL = "vector_tconv_ceil"
    TCAST_TRUNC = "vector_tconv_trunc"
    TCAST_ROUNDING = "vector_tconv_rounding"
    REDUCE_INIT = "vector_reduce_init"
    REDUCE_SUM = "vector_reduce_sum"
    REDUCE_MIN = "vector_reduce_min"
    REDUCE_MAX = "vector_reduce_max"
    REDUCE_ARGMIN = "vector_reduce_argmin"
    REDUCE_ARGMAX = "vector_reduce_argmax"
    REDUCE = "vector_reduce"
    SELECT_EQ = "vector_select_eq"
    SELECT_NE = "vector_select_ne"
    SELECT_GT = "vector_select_gt"
    SELECT_GE = "vector_select_ge"
    SELECT_LT = "vector_select_lt"
    SELECT_LE = "vector_select_le"
    SELECT = "vector_select_bool"
    SELECTVS = "vector_selects_bool"
    AUTO = "vector_auto"
    PHONY_INSN = "phony_insn"
    IM2COL = "im2col"
    SET_FMATRIX = "set_fmatrix"
    MAD = "mad"
    DEPTHWISE_CONV = "depthwise_conv"
    DMA_COPY = "dma_copy"
    DMA_PADDING = "dma_padding"
    DATA_MOV = "data_mov"
    SCALAR = "scalar"
    SCALAR_SQRT = "scalar_sqrt"
    VSCSPLIT = "vscsplit"

    ASCEND_310 = platform.ASCEND_310
    ASCEND_310B = platform.ASCEND_310B
    ASCEND_910 = platform.ASCEND_910
    ASCEND_910H = platform.ASCEND_910H
    ASCEND_910M = platform.ASCEND_910M
    ASCEND_910P = platform.ASCEND_910P
    HI3796CV300ES = platform.HI3796CV300ES
    HI3796CV300CS = platform.HI3796CV300CS
    SD3403 = platform.SD3403
    ASCEND_610 = platform.ASCEND_610
    ASCEND_310P = platform.ASCEND_310P
    BS9SX1A = platform.BS9SX1A
    ASCEND_SD = platform.ASCEND_SD
    AIC_310P = platform.AIC_310P
    VEC_310P = platform.VEC_310P
    AIC_610 = platform.AIC_610
    VEC_610 = platform.VEC_610
    HI3796CV300ESAIC = platform.HI3796CV300ESAIC
    HI3796CV300CSAIC = platform.HI3796CV300CSAIC
    SD3403AIC = platform.SD3403AIC
    ASCEND_SD_AIC = platform.ASCEND_SD_AIC
    scope_cbuf = platform.scope_cbuf
    scope_ubuf = platform.scope_ubuf
    scope_ca = platform.scope_ca
    scope_cb = platform.scope_cb
    scope_cc = platform.scope_cc
    scope_reg = platform.scope_reg
    scope_vreg = platform.scope_vreg
    scope_preg = platform.scope_preg
    scope_areg = platform.scope_areg
    scope_ureg = platform.scope_ureg
    scope_wreg = platform.scope_wreg
    scope_aicpu = platform.scope_aicpu
    scope_gm = platform.scope_gm
    scope_cbuf_fusion = platform.scope_cbuf_fusion
    scope_smask = platform.scope_smask
    dma_copy = platform.dma_copy
    dma_copy_global = platform.dma_copy_global

    SOC_VERSION = platform.SOC_VERSION
    SHORT_SOC_VERSION = platform.SHORT_SOC_VERSION
    FULL_SOC_VERSION = platform.FULL_SOC_VERSION
    AICORE_TYPE = platform.AICORE_TYPE
    CORE_NUM = platform.CORE_NUM
    UB_SIZE = platform.UB_SIZE
    L2_SIZE = platform.L2_SIZE
    L1_SIZE = platform.L1_SIZE
    CUBE_SIZE = platform.CUBE_SIZE
    L0A_SIZE = platform.L0A_SIZE
    L0B_SIZE = platform.L0B_SIZE
    L0C_SIZE = platform.L0C_SIZE
    SMASK_SIZE = platform.SMASK_SIZE
    UNZIP = platform.UNZIP
    CUBE_VECTOR_SPLIT = platform.CUBE_VECTOR_SPLIT
    COMPILER_ARCH = platform.COMPILER_ARCH

    intrinsic_check_support = platform.intrinsic_check_support
    VECTOR_INST_BLOCK_NUM = platform.VECTOR_INST_BLOCK_NUM
    VECTOR_INST_BLOCK_WIDTH = platform.VECTOR_INST_BLOCK_WIDTH
    VECTOR_INST_MAX_REPEAT_TIMES = platform.VECTOR_INST_MAX_REPEAT_TIMES
    BLOCK_IN = platform.BLOCK_IN
    BLOCK_OUT = platform.BLOCK_OUT
    BLOCK_REDUCE = platform.BLOCK_REDUCE
    BLOCK_REDUCE_INT8 = platform.BLOCK_REDUCE_INT8
    BLOCK_VECTOR = platform.BLOCK_VECTOR
    GEMM_MODE = platform.GEMM_MODE
    CONV_MODE = platform.CONV_MODE
    C0_SIZE = platform.C0_SIZE
    ELEMENTS_VECTOR_OP_FP16 = platform.ELEMENTS_VECTOR_OP_FP16
    CUBE_MKN = platform.CUBE_MKN
    CCE_AXIS = cce_util.CCE_AXIS

    fusion_manager = tbe_fusion_manager

    get_soc_spec = platform_info.get_soc_spec
    api_check_support = platform_info.api_check_support
    get_bit_len = platform_info.get_bit_len
    get_align_factor = platform_info.get_align_factor


tbe = platform_tbe.dsl
para_check = platform_tbe.common.utils.para_check
shape_util = platform_tbe.common.utils.shape_util
error_manager = platform_tbe.common.utils.errormgr
op_tiling = platform_tbe.common.utils.op_tiling

error_manager_vector = platform_tbe.common.utils.errormgr
error_manager_util = platform_tbe.common.utils.errormgr.error_manager_util
error_manager_cube = platform_tbe.common.utils.errormgr.error_manager_cube
classify = tbe.classify
operation = tbe_operation
tik = tbe_tik
tvm = tbe_tvm
tbe_context = platform_tbe.common.context
log = tbe_log
# pylint: disable=invalid-name
tbe_platform = PlatformApi
tuple_sum = te_tuple_sum
