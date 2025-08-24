/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/
#ifndef GRAPH_ASCEND_REG_OPS_H
#define GRAPH_ASCEND_REG_OPS_H

#include <type_traits>
#include <utility>
#include "graph/operator_reg.h"
#include "ascend_ir/ascend_ir_core/ascend_ir.h"

#define OP_END_FACTORY_REG_WITHOUT_REGISTER(x) __OP_END_IMPL_WITHOUT_REGISTER__(x)

#endif  // GRAPH_ASCEND_REG_OPS_H
