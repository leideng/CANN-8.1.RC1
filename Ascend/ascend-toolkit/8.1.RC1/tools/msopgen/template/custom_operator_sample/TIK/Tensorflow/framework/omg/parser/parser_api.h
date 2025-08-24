/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_OMG_PARSER_PARSER_API_H_
#define INC_FRAMEWORK_OMG_PARSER_PARSER_API_H_

#include <map>
#include <string>
#include "external/ge/ge_api_error_codes.h"
#include "graph/ascend_string.h"

namespace ge {
// Initialize parser
GE_FUNC_VISIBILITY Status ParserInitialize(const std::map<std::string, std::string>& options);
// Finalize parser, release all resources
GE_FUNC_VISIBILITY Status ParserFinalize();
// Initialize parser
GE_FUNC_VISIBILITY Status ParserInitialize(const std::map<ge::AscendString, ge::AscendString>& options);
}  // namespace ge
#endif // INC_FRAMEWORK_OMG_PARSER_PARSER_API_H_
