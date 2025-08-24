/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "ge_graph_dumper.h"

namespace ge {
namespace {
struct DefaultDumper : public GeGraphDumper {
  void Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix) override {
    (void)graph;
    (void)suffix;
  }
};
DefaultDumper default_dumper;
GeGraphDumper *register_checker = &default_dumper;
}

GeGraphDumper &GraphDumperRegistry::GetDumper() {
  return *register_checker;
}
void GraphDumperRegistry::Register(GeGraphDumper &dumper) {
  register_checker = &dumper;
}
void GraphDumperRegistry::Unregister() {
  register_checker = &default_dumper;
}
}  // namespace ge
