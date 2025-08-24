/**
 * Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layout.h
 * \brief
 */

#ifndef INCLUDE_LAYOUT_H
#define INCLUDE_LAYOUT_H

enum class DataFormatT { 
    ND = 0,
    NZ,
    ZN,
    ZZ,
    NN,
    VECTOR
}; 

#endif // INCLUDE_LAYOUT_H