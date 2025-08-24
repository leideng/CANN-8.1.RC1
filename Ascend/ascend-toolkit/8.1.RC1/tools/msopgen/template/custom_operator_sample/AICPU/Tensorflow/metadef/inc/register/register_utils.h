/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef REGISTER_REGISTER_UTILS_H
#define REGISTER_REGISTER_UTILS_H

#include <google/protobuf/message.h>
#include "external/register/register_types.h"
#include "external/register/register_error_codes.h"
#include "external/register/register.h"
#include "external/graph/operator.h"

namespace domi {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OperatorAutoMapping(const Message *op_src, ge::Operator &op);
}  // namespace domi
#endif  // REGISTER_REGISTER_UTILS_H
