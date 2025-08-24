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
 * \file kernel_operator_vec_brcb_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BRCB_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_BRCB_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_brcb.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_brcb_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_brcb_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_brcb_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_brcb_impl.h"
#endif
#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup brcb Level 0
 * @brief this function fetches 8 b16/b32 data from src0Local, broadcast each data into one 32B block,
 * @brief then finally writes these 8 blocks into dstLocal continously.
 * @brief gather element in the uint of block
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Brcb(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_BRCB_INTERFACE_H