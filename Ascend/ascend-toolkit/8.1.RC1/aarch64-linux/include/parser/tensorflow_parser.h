/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_ACL_PARSER_TENSORFLOW_H_
#define INC_EXTERNAL_ACL_PARSER_TENSORFLOW_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"

namespace ge {
PARSER_FUNC_VISIBILITY graphStatus aclgrphParseTensorFlow(const char *model_file, ge::Graph &graph);
PARSER_FUNC_VISIBILITY graphStatus aclgrphParseTensorFlow(const char *model_file,
    const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph);
}  // namespace ge

#endif  // INC_EXTERNAL_ACL_PARSER_TENSORFLOW_H_
