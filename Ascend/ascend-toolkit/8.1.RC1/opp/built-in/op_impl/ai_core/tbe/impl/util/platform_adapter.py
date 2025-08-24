#!/usr/bin/python
# -*- coding: utf-8 -*-
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
from typing import Any
from typing import Dict
from typing import Optional

# 'pylint: disable=unused-import,invalid-name,reimported
from tbe.dsl.base import operation as tbe_operation
from tbe.dsl.api import gemm
from tbe.dsl.compute.mmad_compute import get_matmul_performance_format
import tbe as platform_tbe
import tbe.common.buildcfg as tbe_build
import tbe.common.register as tbe_register
from tbe import tik as tbe_tik
from tbe import tvm as tbe_tvm
from tbe.common import platform
from tbe.common import buildcfg
from tbe.common.buildcfg import build_config
from tbe.dsl.instrinsic.cce_util import get_const
from tbe.common.buildcfg import get_current_build_config
from tbe.common.buildcfg.default_buildcfg import dynamic_build_config_dict
from tbe.common.platform import platform_info
from tbe.common.utils import log as tbe_log
from tbe.common.utils.errormgr import error_manager_vector
from tbe.common.utils.errormgr import error_manager_cube
from tbe.common.utils import para_check
from tbe.common.rl_bank import rl_bank
from tbe.common.rl_bank import bank_manager
from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_SPARSE_4_2
from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_UNZIP
from tbe.common.utils.op_util.op_util_conv2d import COMPRESS_ALG_SUPPORT
from tbe.common.utils.op_util.op_util_conv2d import CUBE_MKN_IDX_K
from tbe.common.utils.op_util.op_util_conv2d import CUBE_MKN_IDX_N
from te.platform.fusion_manager import fusion_manager as tbe_fusion_manager
from te.lang.cce import tuple_sum as te_tuple_sum
from impl.util.norm_pattern_adapter import NormPattern
from tbe.common.platform import set_current_compile_soc_info
from tbe.dsl.compute.conv_compute import conv_compress
from tbe.common.platform import intrinsic_check_support
from tbe.dsl import auto_schedule
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import in_record
from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp
from tbe.dsl.instrinsic.cce_intrin_md import reset_mask_insn
from tbe.dsl.instrinsic.cce_util import get_const
from tbe.dsl.instrinsic.cce_util import get_type_bits
from tbe.dsl.instrinsic import cce_emitinsn_params
from tbe.common.context import op_context
from tbe.dsl.compute.common import tf_get_windowed_output_size_verbose_v2
from tbe.dsl.instrinsic import cce_util
from tbe.dsl.compute.max_pool2d_3_2_fusion_compute import max_pool_compute
from tbe.dsl.compute.util import is_cast_support
from tbe.dsl.compute.cast import _cast as inernal_cast
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
from tbe.dsl.compute.common import tf_get_windowed_output_size_verbose
from impl.util.compute.scatter import Scatter


# 'pylint: disable=too-few-public-methods, too-many-instance-attributes, invalid-name
class OpPatternMode:
    """
    op pattern mode
    """
    NONE = ""
    GATHER = "gather"
    ELEWISE = "elewise"
    ELEWISE_WITH_BROADCAST = "broadcast"
    REDUCE = "reduce"
    TRANSDATA = "transdata"
    NORM = NormPattern.PATTERN_NAME
    TUPLE_REDUCE = "tuple_reduce"
    POOLING_GRAD_WITH_ARG = "pooling_grad_with_arg"
    POOLING = "pooling"
    ASCEND_ANTI_QUANT = "anti_quant"
    ASCEND_QUANT = "quant"
    SEGMENT = "segment"
    SPARSE_APPLY = "sparse_apply"
    SORT = "sort"
    CONV2D_BP_FILTER = "Conv2d_backprop_filter"


class OpImplMode:
    """
    op implement mode high_performance or high_precision
    """
    HIGH_PERFORMANCE = "high_performance"
    HIGH_PRECISION = "high_precision"
    SUPER_PERFORMANCE = "super_performance"
    NORM_CLASS = "norm_class"
    SUPPORT_OUT_OF_BOUND_INDEX = "support_out_of_bound_index"


class OpTbeImplMode:
    """
    tbe implement mode high_precision
    """
    TBE_HIGH_PRECISION = "high_precision"


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
    AS31XM1 = platform.AS31XM1
    ASCEND_910 = platform.ASCEND_910
    ASCEND_910H = platform.ASCEND_910H
    ASCEND_910M = platform.ASCEND_910M
    ASCEND_910P = platform.ASCEND_910P
    HI3796CV300ES = platform.HI3796CV300ES
    HI3796CV300CS = platform.HI3796CV300CS
    SD3403 = platform.SD3403
    ASCEND_610 = platform.ASCEND_610
    BS9SX1A = platform.BS9SX1A
    ASCEND_310P = platform.ASCEND_310P
    ASCEND_SD = platform.ASCEND_SD
    AIC_310P = platform.AIC_310P
    VEC_310P = platform.VEC_310P
    AIC_610 = platform.AIC_610
    VEC_610 = platform.VEC_610
    HI3796CV300ESAIC = platform.HI3796CV300ESAIC
    HI3796CV300CSAIC = platform.HI3796CV300CSAIC
    SD3403AIC = platform.SD3403AIC
    ASCEND_SD_AIC = platform.ASCEND_SD_AIC
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
    intrinsic_check_support = platform.intrinsic_check_support

    fusion_manager = tbe_fusion_manager

    get_soc_spec = platform_info.get_soc_spec
    api_check_support = platform_info.api_check_support
    get_bit_len = platform_info.get_bit_len
    get_align_factor = platform_info.get_align_factor
    get_block_size = platform_info.get_block_size


tbe = platform_tbe.dsl
tbe_context = platform_tbe.common.context
op_tiling = platform_tbe.common.utils.op_tiling
para_check = platform_tbe.common.utils.para_check
shape_util = platform_tbe.common.utils.shape_util
error_manager = platform_tbe.common.utils.errormgr
error_manager_util = platform_tbe.common.utils.errormgr.error_manager_util
error_manager_cube = platform_tbe.common.utils.errormgr.error_manager_cube
tik = tbe_tik
tvm = tbe_tvm
log = tbe_log
tuple_sum = te_tuple_sum
classify = tbe.classify
operation = tbe_operation
register_operator = tbe_register.register_operator
tbe_platform = PlatformApi


def tbe_classify(ins: list, mode: str, extra_params: Optional[Dict[str, Any]] = None):
    """
    register op compute func

    Parameters
    ----------
    ins : list
        list of input dict
    mode: string
        classfy mode, [OpPatternMode.ELEWISE, OpPatternMode.ELEWISE_WITH_BROADCAST,
                       OpPatternMode.reduce, OpPatternMode.TRANSDATA, OpPatternMode.NORM]
    extra_params: dict
        extra_params

    Returns
    -------
    inc : list
        list of classify result
    """
    from impl.util import util_common
    for i, input_dict in enumerate(ins):
        # not dict type will continue
        if not isinstance(input_dict, dict):
            continue
        # dict is dynamic will continue
        if util_common.is_unknown(input_dict):
            continue
        # dict is static and do not have range, will set range with shape
        if input_dict is not None and not input_dict.get("range"):
            input_shape = input_dict.get("shape")
            if input_shape is not None:
                input_range = []
                for _, dim in enumerate(input_shape):
                    input_range.append((dim, dim))
                ins[i]["range"] = input_range

    return tbe.classify(ins, mode, extra_params)


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


def check_supported_vcopy():
    """
    vcopy support after ascend910B

    Returns
    -------
    True or False
    """
    return tbe_platform.api_check_support("tik.vcopy")


def check_support_block_size_16():
    """
    For nano series chips, one of the important features is that the block size is 16 bytes.
    One of the important conditions for nano chips.

    Returns
    -------
    True or False
    """
    block_size = tbe_platform.get_block_size()
    return True if block_size == 16 else False
