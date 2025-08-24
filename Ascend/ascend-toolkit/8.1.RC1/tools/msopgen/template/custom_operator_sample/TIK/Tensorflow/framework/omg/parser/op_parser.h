/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_OMG_PARSER_OP_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_OP_PARSER_H_

#include <google/protobuf/text_format.h>
#include "framework/omg/parser/parser_types.h"
#include "framework/omg/omg_inner_types.h"
#include "proto/om.pb.h"
#include "graph/utils/op_desc_utils.h"

using google::protobuf::Message;

namespace ge {
/**
 * @ingroup domi_omg
 * @brief Used to analyze operator information
 *
 */
class GE_FUNC_VISIBILITY OpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief Deconstructor
   */
  virtual ~OpParser() {}

  /**
   * @ingroup domi_omg
   * @brief Analytic operator parameters
   * @param [in] op_src Parameter data to be resolved
   * @param [out] graph Parsed parameter data
   * @return SUCCESS
   * @return FAILED
   */
  virtual domi::Status ParseParams(const google::protobuf::Message *op_src, ge::OpDescPtr &op_desc) = 0;

  /**
   * @ingroup domi_omg
   * @brief Analytic operator parameters
   * @param [in] op_src Parameter data to be resolved
   * @param [out] Operator parameter data
   * @return SUCCESS
   * @return FAILED
   */
  virtual domi::Status ParseParams(const google::protobuf::Message *op_src, ge::Operator &op_dest) = 0;

  /**
   * @ingroup domi_omg
   * @brief Analytic operator weight information
   * @param [in] op_src Weight data to be resolved
   * @param [out] op_dest Weight data after analysis
   * @return SUCCESS
   * @return FAILED
   */
  virtual domi::Status ParseWeights(const google::protobuf::Message *op_src, ge::NodePtr &node) = 0;

  /**
   * @ingroup domi_omg
   * @brief Get the format information according to the parameters in the operator
   * @param [in] op_src Parameter data to be resolved
   * @param [out] format Output the parsed format
   * @return SUCCESS
   * @return FAILED
   */
  virtual domi::Status GetFormat(const google::protobuf::Message *op_src, domi::domiTensorFormat_t &format) {
    (void)op_src;
    // Indicates that the op does not provide a value for format
    format = domi::DOMI_TENSOR_RESERVED;
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_PARSER_OP_PARSER_H_
