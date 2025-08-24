/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file service_quant_matmul.h
 * \brief
 */

#ifndef SERVICE_QUANT_MATMUL_H
#define SERVICE_QUANT_MATMUL_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

template <typename T, typename O, typename MMType>
__aicore__ inline void QuantMatmul(MMType&mm, const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                                   const GlobalTensor<O>& tensorCGm, GlobalTensor<O>& tensorQuantScaleGm) {

    // L1addr, L1Size,
    // L0AAddr, // baseM*baseK
    // L0BAddr, // baseN*baseK
    // L0CAddr,

}

}

#endif