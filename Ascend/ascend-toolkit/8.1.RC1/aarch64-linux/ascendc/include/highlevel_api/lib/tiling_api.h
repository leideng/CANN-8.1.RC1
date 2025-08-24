/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_api.h
 * \brief
 */
#ifndef LIB_TILING_API_H
#define LIB_TILING_API_H
#include "matmul/matmul_tiling.h"
#include "matmul/bmm_tiling.h"
#include "activation/softmax_tiling.h"
#include "activation/logsoftmax_tiling.h"
#include "filter/dropout_tiling.h"
#include "sort/sort_tiling_intf.h"
#include "index/arithprogression_tiling.h"
#include "quantization/ascend_dequant_tiling.h"
#include "quantization/ascend_quant_tiling.h"
#include "quantization/ascend_antiquant_tiling.h"
#include "reduce/sum_tiling.h"
#include "activation/silu_tiling.h"
#include "activation/swish_tiling.h"
#include "activation/gelu_tiling.h"
#include "pad/pad_tiling.h"
#include "normalization/rmsnorm_tiling.h"
#include "normalization/deepnorm_tiling.h"
#include "normalization/layernorm_tiling.h"
#include "normalization/normalize_tiling.h"
#include "normalization/batchnorm_tiling.h"
#include "normalization/layernorm_grad_tiling.h"
#include "normalization/layernorm_grad_beta_tiling.h"
#include "normalization/welfordfinalize_tiling.h"
#include "transpose/confusion_transpose_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "sort/topk_tiling.h"
#include "math/tanh_tiling.h"
#include "activation/sigmoid_tiling.h"
#include "math/frac_tiling.h"
#include "math/acos_tiling.h"
#include "math/asin_tiling.h"
#include "math/acosh_tiling.h"
#include "math/asinh_tiling.h"
#include "math/sin_tiling.h"
#include "math/cos_tiling.h"
#include "math/atan_tiling.h"
#include "math/power_tiling.h"
#include "math/log_tiling.h"
#include "math/cosh_tiling.h"
#include "math/clamp_tiling.h"
#include "math/erf_tiling.h"
#include "math/erfc_tiling.h"
#include "math/round_tiling.h"
#include "math/sinh_tiling.h"
#include "activation/swiglu_tiling.h"
#include "math/tan_tiling.h"
#include "select/selectwithbytesmask_tiling.h"
#include "math/trunc_tiling.h"
#include "activation/geglu_tiling.h"
#include "math/lgamma_tiling.h"
#include "math/digamma_tiling.h"
#include "math/atanh_tiling.h"
#include "math/xor_tiling.h"
#include "math/sign_tiling.h"
#include "reduce/mean_tiling.h"
#include "math/exp_tiling.h"
#include "math/axpy_tiling.h"
#include "math/ceil_tiling.h"
#include "math/floor_tiling.h"
#include "activation/reglu_tiling.h"
#include "pad/broadcast_tiling.h"
#include "reduce/reduce_xor_sum_tiling.h"
#include "math/cumsum_tiling.h"
#include "math/fmod_tiling.h"
#include "normalization/groupnorm_tiling.h"
#include "hccl/hccl_tilingdata.h"
#include "hccl/hccl_tiling.h"
#endif // LIB_TILING_API_H
