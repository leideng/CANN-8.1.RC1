/**
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
* \file cube_in_buffer.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_H

#include "cube_in_buffer_normal.h"
#include "cube_in_buffer_single_buffer.h"
#include "cube_in_buffer_single_global_buffer.h"
#include "cube_in_buffer_double_buffer.h"
#include "cube_in_buffer_double_global_buffer.h"
#if __CCE_AICORE__ == 220
#include "cube_in_buffer_double_buffer_sparse.h"
#endif

#endif // _CUBE_IN_BUFFER_H_