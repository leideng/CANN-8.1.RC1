/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file sort.h
 * \brief
 */
#ifndef LIB_SORT_SORT_H
#define LIB_SORT_SORT_H

#include "kernel_operator.h"

/*
 * @ingroup Sort
 * @brief Sort them according to the value
 * @param [out] dstLocal output LocalTensor
 * @param [in] concatLocal input LocalTensor
 * @param [in] indexLocal input LocalTensor
 * @param [in] tmpLocal tmp buffer
 * @param [in] repeatTimes repeat times
 * 
 * template <typename T, bool isFullSort>
 * __aicore__ inline void Sort(const LocalTensor<T> &dstLocal, const LocalTensor<T> &concatLocal,
 *     const LocalTensor<uint32_t> &indexLocal, LocalTensor<T> &tmpLocal, const int32_t repeatTimes);
 */

#endif  // LIB_SORT_SORT_H