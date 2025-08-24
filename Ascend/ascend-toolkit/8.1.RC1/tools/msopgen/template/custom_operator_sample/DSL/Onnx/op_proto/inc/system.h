/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file system.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SYSTEM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SYSTEM_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief The main function of the advcance_step operator is to advance the inference step in vLLM, that is,
* update the model status and generate new inputTokens, inputPostions, seqLen, and slotMapping in each generation step.
* Improves the efficiency of vLLM inference. \n

*@par Inputs:
* Six inputs, including:
*@li input_tokens: A 1-D input tensor, and length equal to num_seqs. Must be int64 type. Format is ND.
*@li sampled_token_ids: A 2-D input tensor, which the first dim equal to num_queries and the second dim equal to one.
* Must be int64 type. Format is ND.
*@li input_positions: A 1-D input tensor, and length equal to num_seqs. Must be int64 type. Format is ND.
*@li seq_lens: A 1-D input tensor, and length equal to num_seqs. Must be int64 type. Format is ND.
*@li slot_mapping: A 1-D input tensor, and length equal to num_seqs. Must be int64 type. Format is ND.
*@li block_tables: A 1-D input tensor, and length equal to num_seqs. Must be int64 type. Format is ND. \n

*@par Attributes:
*@li num_seqs: A required Int, which equal to the length of input_tokens, input_positions, seq_lens,
* slot_mapping and block_tables. The value of it must bigger than 0.
*@li num_queries: A required Int, which equal to the length of sampled_token_ids's first dim.
* The value of it must bigger than 0.
*@li block_size: A required Int, which means the basic block length of each block. The value of it must bigger than 0. \n

*@par Outputs:
*@li input_tokens: A 1-D output tensor. The input tensor input_tokens will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li input_positions: A 1-D output tensor. The input tensor input_positions will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li seq_lens: A 1-D output tensor. The input tensor seq_lens will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li slot_mapping: A 1-D output tensor. The input tensor slot_mapping will self-updating and save as itself.
* Must be int64 type. Format is ND. \n

*/

REG_OP(AdvanceStep)
    .INPUT(input_tokens, TensorType({DT_INT64}))
    .INPUT(sampled_token_ids, TensorType({DT_INT64}))
    .INPUT(input_positions, TensorType({DT_INT64}))
    .INPUT(seq_lens, TensorType({DT_INT64}))
    .INPUT(slot_mapping, TensorType({DT_INT64}))
    .INPUT(block_tables, TensorType({DT_INT64}))
    .OUTPUT(input_tokens, TensorType({DT_INT64}))
    .OUTPUT(input_positions, TensorType({DT_INT64}))
    .OUTPUT(seq_lens, TensorType({DT_INT64}))
    .OUTPUT(slot_mapping, TensorType({DT_INT64}))
    .REQUIRED_ATTR(num_seqs, Int)
    .REQUIRED_ATTR(num_queries, Int)
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(AdvanceStep)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SYSTEM_H_
