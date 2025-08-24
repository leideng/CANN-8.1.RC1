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
 * \file stub_reg.h
 * \brief
 */
#ifndef ASCENDC_STUB_REG_H
#define ASCENDC_STUB_REG_H
#include "intri_fun.h"
#include "intri_fmt.h"

namespace AscendC {
extern const int SYM_LEN_MAX;
void StubReg(IntriTypeT type, const char* stub);
void StubInit(void);
} // namespace AscendC
#endif // ASCENDC_STUB_REG_H
