/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_operator_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_INTERFACE_H
#include "kernel_prof_trace_intf.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_dump_tensor_intf.h"
#include "kernel_operator_mm_intf.h"
#include "kernel_operator_gemm_intf.h"
#include "kernel_operator_fixpipe_intf.h"
#include "kernel_operator_conv2d_intf.h"
#include "kernel_operator_common_intf.h"
#include "kernel_operator_proposal_intf.h"
#include "kernel_operator_determine_compute_sync_intf.h"
#include "kernel_operator_vec_transpose_intf.h"
#include "kernel_operator_vec_gather_intf.h"
#include "kernel_operator_vec_scatter_intf.h"
#include "kernel_operator_vec_brcb_intf.h"
#include "kernel_operator_vec_binary_intf.h"
#include "kernel_operator_vec_binary_scalar_intf.h"
#include "kernel_operator_vec_cmpsel_intf.h"
#include "kernel_operator_vec_duplicate_intf.h"
#include "kernel_operator_vec_reduce_intf.h"
#include "kernel_operator_vec_gather_mask_intf.h"
#include "kernel_operator_vec_mulcast_intf.h"
#include "kernel_operator_vec_bilinearinterpalation_intf.h"
#include "kernel_operator_vec_createvecindex_intf.h"
#include "kernel_operator_vec_ternary_scalar_intf.h"
#include "kernel_operator_vec_unary_intf.h"
#include "kernel_operator_vec_vconv_intf.h"
#include "kernel_operator_vec_vpadding_intf.h"
#include "kernel_operator_scalar_intf.h"
#include "kernel_operator_sys_var_intf.h"

#include "lib/kernel_api.h"

#if __CCE_AICORE__ == 220
#include "core_mng/roc/kernel_operator_cube_group_intf.h"
#include "core_mng/roc/kernel_operator_group_barrier_intf.h"
#endif

#if (!defined(__DAV_M310__))
#include "lib/filter/dropout.h"
#include "lib/activation/sigmoid.h"
#include "lib/activation/softmax.h"
#include "lib/activation/simplesoftmax.h"
#include "lib/activation/softmaxflashv2.h"
#include "lib/activation/softmaxgrad.h"
#endif
#if __CCE_AICORE__ < 300
#include "lib/index/arithprogression.h"
#include "lib/normalization/layernormgrad.h"
#include "lib/normalization/layernormgradbeta.h"
#include "lib/pad/pad.h"
#include "lib/math/frac.h"
#include "lib/math/power.h"
#include "lib/math/log.h"
#include "lib/math/sin.h"
#include "lib/math/cos.h"
#include "lib/math/asin.h"
#include "lib/math/acos.h"
#include "lib/math/asinh.h"
#include "lib/math/acosh.h"
#include "lib/math/atan.h"
#include "lib/math/cosh.h"
#include "lib/math/erf.h"
#include "lib/math/erfc.h"
#include "lib/math/clamp.h"
#include "lib/normalization/rmsnorm.h"
#include "lib/normalization/batchnorm.h"
#include "lib/math/tanh.h"
#include "lib/math/atanh.h"
#include "lib/normalization/deepnorm.h"
#include "lib/math/exp.h"
#include "lib/normalization/layernorm.h"
#include "lib/reduce/sum.h"
#include "lib/activation/silu.h"
#include "lib/activation/gelu.h"
#include "lib/quantization/ascend_quant.h"
#include "lib/quantization/ascend_dequant.h"
#include "lib/quantization/ascend_antiquant.h"
#include "lib/activation/logsoftmax.h"
#include "lib/activation/softmaxflash.h"
#include "lib/transpose/confusion_transpose.h"
#include "lib/select/selectwithbytesmask.h"
#include "lib/math/sinh.h"
#include "lib/activation/swiglu.h"
#include "lib/activation/reglu.h"
#include "lib/math/tan.h"
#include "lib/math/round.h"
#include "lib/math/trunc.h"
#include "lib/activation/swish.h"
#include "lib/sort/topk.h"
#include "lib/activation/geglu.h"
#include "lib/math/lgamma.h"
#include "lib/math/digamma.h"
#include "lib/math/xor.h"
#include "lib/math/sign.h"
#include "lib/reduce/mean.h"
#include "lib/math/axpy.h"
#include "lib/math/ceil.h"
#include "lib/math/floor.h"
#include "lib/pad/broadcast.h"
#include "lib/reduce/reduce_xor_sum.h"
#include "lib/math/cumsum.h"
#endif
#endif // ASCENDC_MODULE_OPERATOR_INTERFACE_H
