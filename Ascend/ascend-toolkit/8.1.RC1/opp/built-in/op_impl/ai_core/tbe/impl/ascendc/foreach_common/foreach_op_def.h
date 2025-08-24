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
 * \file foreach_op_def.h
 * \brief
 */
#ifndef __FOR_EACH_OP_DEF_H__
#define __FOR_EACH_OP_DEF_H__
namespace ForeachOpDef {

/* foreach OP info, device and host share OPIDs */
// add user OPID begin
#define ADD_TENSOR_LIST 1
#define ADD_TENSOR_LIST_WITH_SCALAR 2
#define ADDS_TENSOR_LIST_AND_SCALAR 3
#define ADDCMUL_TENSOR_LIST 4
#define ADDCMUL_TENSOR_LIST_WITH_SCALAR 5
#define EXP_TENSOR_LIST 6
#define ABS_TENSOR_LIST 7

// add user OPID end
#define OPCODE_LAST 128
#define END_OPCODE ((OPCODE_LAST) + 1)

}  // namespace ForeachOpDef
#endif  // __FOR_EACH_OP_DEF_H__