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
* \file bias_scheduler.h
* \brief
*/

#ifndef IMPL_MATMUL_SCHEDULER_BIAS_BIAS_SCHEDULER_H
#define IMPL_MATMUL_SCHEDULER_BIAS_BIAS_SCHEDULER_H

#if __CCE_AICORE__ <= 200
#include "bias_scheduler_v200.h"
#else
#include "bias_scheduler_batch.h"
#include "bias_scheduler_v220.h"
#endif
#endif // _BIAS_SCHEDULER_H_
