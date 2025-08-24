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
 * \file kernel_struct_vdeq.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_VDEQ_H
#define ASCENDC_MODULE_STRUCT_VDEQ_H

namespace AscendC {
struct VdeqInfo {
    __aicore__ VdeqInfo() {}
    
    __aicore__ VdeqInfo(const float vdeqScaleIn[VDEQ_TENSOR_SIZE], const int16_t vdeqOffsetIn[VDEQ_TENSOR_SIZE],
        const bool vdeqSignModeIn[VDEQ_TENSOR_SIZE])
    {
        for (int32_t i = 0; i < VDEQ_TENSOR_SIZE; ++i) {
            vdeqScale[i] = vdeqScaleIn[i];
            vdeqOffset[i] = vdeqOffsetIn[i];
            vdeqSignMode[i] = vdeqSignModeIn[i];
        }
    }

    float vdeqScale[VDEQ_TENSOR_SIZE] = { 0 };
    int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = { 0 };
    bool vdeqSignMode[VDEQ_TENSOR_SIZE] = { 0 };
};
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_VDEQ_H