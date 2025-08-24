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
 * \file intri_fun.h
 * \brief
 */
#ifndef ASCENDC_INTRI_FUN_H
#define ASCENDC_INTRI_FUN_H
#include <cstdint>
#include "stub_def.h"
#include "stub_fun.h"

#define INTRI_FUN_NUM 3154

typedef enum {
    INTRI_TYPE_CPU = 0,
    INTRI_TYPE_NPU,
    INTRI_TYPE_CCE,
    INTRI_TYPE_USER1,
    INTRI_TYPE_USER2,
    INTRI_TYPE_MAX
} IntriTypeT;
using PfIntriFun = uint64_t (*) (...);
void IntriFunAdd(int32_t fid, IntriTypeT type, PfIntriFun fun);
#endif // ASCENDC_INTRI_FUN_H
