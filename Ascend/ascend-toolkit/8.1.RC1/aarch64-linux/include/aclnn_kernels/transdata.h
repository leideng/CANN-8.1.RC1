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
#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_TRANSDATA_H_
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_TRANSDATA_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *ReFormat(const aclTensor *x, const op::Format &format, aclOpExecutor *executor=nullptr);

/**
 * TransData
 * Formal Transdata. Set the c0 size strictly based on the data type and chip block size.
 * support data type as follows: fp16,fp32,int32,uint32,int8,uint8
 * fp16: block_size/2
 * fp32/int32/uint32: block_size/4 (this is different from `TransDataSpecial`)
 * int8/uint8: block_size/1
 *
 * @param x : aclTensor need to transpose
 * @param dstPrimaryFormat: dstPrimaryFormat like NC1HWC0
 * @param groups: groups
 * @param executor: executor should not be null
 * @return trans format tensor
 */
const aclTensor *TransData(const aclTensor *x,
                           op::Format dstPrimaryFormat,
                           int64_t groups,
                           aclOpExecutor *executor);
/**
 * Special Transdata. Set the c0 size strictly based on the data type and chip block size.
 * this transdata c0 size rule:
 *     fp16: block_size/2
 *     fp32/int32/uint32: block_size/2
 *     int8/uint8: block_size/1
 * bool not supported, should do:
 *     (NCHW, bool)-> cast -> (NCHW, fp16) -> TransDataSpecial -> (5HD, fp16) -> cast -> (5HD, bool)
 *     (5HD, bool)-> cast -> (5HD, fp16) -> TransDataSpecial -> (NCHW, fp16) -> cast -> (NCHW, bool)
 *
 * @param x : aclTensor need to transpose
 * @param dstPrimaryFormat: dstPrimaryFormat like NC1HWC0
 * @param groups: groups
 * @param executor: executor should not be null
 * @return trans format tensor
 */
const aclTensor *TransDataSpecial(const aclTensor *x,
                                  op::Format dstPrimaryFormat,
                                  int64_t groups,
                                  aclOpExecutor *executor);

}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_TRANSDATA_H_
