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
 * \file kernel_api.h
 * \brief
 */
#ifndef LIB_KERNEL_API_H
#define LIB_KERNEL_API_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ != 310)
#include "filter/dropout.h"
#include "activation/sigmoid.h"
#include "activation/softmax.h"
#include "activation/simplesoftmax.h"
#include "activation/softmaxflashv2.h"
#include "activation/softmaxflashv3.h"
#include "activation/softmaxgrad.h"
#include "math/xor.h"
#include "math/floor.h"
#include "sort/sort.h"
#endif // __CCE_AICORE__ != 310

#include "std/tuple.h"
#include "std/type_traits.h"
#include "std/utility.h"
#include "std/algorithm.h"

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ < 300)
#include "index/arithprogression.h"
#include "normalization/layernormgrad.h"
#include "normalization/layernormgradbeta.h"
#include "pad/pad.h"
#include "hccl/hccl.h"
#include "math/frac.h"
#include "math/power.h"
#include "math/log.h"
#include "math/sin.h"
#include "math/cos.h"
#include "math/asin.h"
#include "math/acos.h"
#include "math/asinh.h"
#include "math/acosh.h"
#include "math/atan.h"
#include "math/cosh.h"
#include "math/erf.h"
#include "math/erfc.h"
#include "math/clamp.h"
#include "normalization/rmsnorm.h"
#include "normalization/batchnorm.h"
#include "math/tanh.h"
#include "math/atanh.h"
#include "normalization/deepnorm.h"
#include "math/exp.h"
#include "normalization/layernorm.h"
#include "normalization/welfordfinalize.h"
#include "normalization/normalize.h"
#include "reduce/sum.h"
#include "activation/silu.h"
#include "activation/gelu.h"
#include "quantization/ascend_quant.h"
#include "quantization/ascend_dequant.h"
#include "quantization/ascend_antiquant.h"
#include "activation/logsoftmax.h"
#include "activation/softmaxflash.h"
#include "transpose/confusion_transpose.h"
#include "select/selectwithbytesmask.h"
#include "math/sinh.h"
#include "activation/swiglu.h"
#include "activation/reglu.h"
#include "math/tan.h"
#include "math/round.h"
#include "math/trunc.h"
#include "activation/swish.h"
#include "sort/topk.h"
#include "activation/geglu.h"
#include "math/lgamma.h"
#include "math/digamma.h"
#include "math/sign.h"
#include "reduce/mean.h"
#include "math/axpy.h"
#include "math/ceil.h"
#include "pad/broadcast.h"
#include "reduce/reduce_xor_sum.h"
#include "math/cumsum.h"
#include "math/fmod.h"
#include "normalization/groupnorm.h"
#include "utils/init_global_memory.h"
#endif // __CCE_AICORE__ < 300

#endif // LIB_KERNEL_API_H
