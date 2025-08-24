/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. 2025
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file add_cv.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_ADD_CV_IMPL__H
#define IMPL_STD_ASCENDC_STD_ADD_CV_IMPL__H

namespace AscendC {
namespace Std {

template <typename Tp>
struct add_cv {
    using type = const volatile Tp;
};

template <typename Tp>
using add_cv_t = typename add_cv<Tp>::type;

}
}

#endif // IMPL_STD_ASCENDC_STD_ADD_CV_IMPL__H
