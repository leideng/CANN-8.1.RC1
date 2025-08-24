/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef __INC_METADEF_OP_DESC_UTILS_EX_H
#define __INC_METADEF_OP_DESC_UTILS_EX_H

#include "graph/op_desc.h"

namespace ge {
class OpDescUtilsEx {
 public:
  // Detach from OpDesc
  static graphStatus CallInferFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFormatFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferValueRangeFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus OpVerify(const OpDescPtr &op_desc);
  static graphStatus InferShapeAndType(const OpDescPtr &op_desc);
  static graphStatus InferDataSlice(const OpDescPtr &op_desc);
  static void SetType(OpDescPtr &op_desc, const std::string &type);
  static void ResetFuncHandle(OpDescPtr &op_desc);
  static void SetTypeAndResetFuncHandle(OpDescPtr &op_desc, const std::string &type);
  static void UpdateShapeAndDType(const GeTensorDescPtr &src, const GeTensorDescPtr &dst);

 private:
  static graphStatus CallInferFuncV1(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFuncV2(const OpDescPtr &op_desc, Operator &op);
  static graphStatus InferShapeByOutputShapesAttr(const OpDescPtr &op_desc);
};
} // namespace ge
#endif // __INC_METADEF_OP_DESC_UTILS_EX_H
