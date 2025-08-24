/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_CHECK_H_
#define INC_EXTERNAL_REGISTER_OP_CHECK_H_

#include "op_check_register.h"
namespace optiling {

#define REG_CHECK_SUPPORT(op_type, func)                                                                               \
  static OpCheckFuncHelper op_check_registry_##op_type##_check_supported(FUNC_CHECK_SUPPORTED, #op_type, func)
#define REG_OP_SELECT_FORMAT(op_type, func)                                                                            \
  static OpCheckFuncHelper op_check_registry_##op_type##_op_select_format(FUNC_OP_SELECT_FORMAT, #op_type, func)
#define REG_OP_SUPPORT_INFO(op_type, func)                                                                             \
  static OpCheckFuncHelper op_check_registry_##op_type##_get_op_support_info(FUNC_GET_OP_SUPPORT_INFO, #op_type, func)
#define REG_OP_SPEC_INFO(op_type, func)                                                                                \
  static OpCheckFuncHelper op_check_registry_##op_type##_get_specific_info(FUNC_GET_SPECIFIC_INFO, #op_type, func)

#define REG_OP_PARAM_GENERALIZE(op_type, generalize_func)                                                              \
  static OpCheckFuncHelper op_check_generalize_registry_##op_type(#op_type, generalize_func)

#define REG_REPLAY_FUNC(op_type, soc_version, func)                                                                    \
  static ReplayFuncHelper op_replay_registry_##op_type_##soc_version(#op_type, #soc_version, func)
}  // end of namespace optiling
#endif  // INC_EXTERNAL_REGISTER_OP_CHECK_H_