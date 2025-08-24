/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_COMMONOPPARAM_H
#define ATB_COMMONOPPARAM_H
#include <cstdint>
#include <acl/acl.h>

//!
//! \file common_op_params.h
//!
//! \brief 定义加速库所有通用算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace common {

//!
//! \struct EventParam
//!
//! \brief 流间同步功能。Record或者Wait Event
//!
//! \warning 需要使用aclrtSetOpWaitTimeout设置等待Event完成的超时时间
//!
struct EventParam {
    //!
    //! \enum OperatorType
    //! \brief OperatorType支持的值
    //!
    enum OperatorType : int {
        UNDEFINED = 0,  //!< 默认值，不做任何操作
        RECORD,         //!< 在Stream中记录一个Event
        WAIT            //!< 阻塞指定Stream的运行，直至指定的Event完成
    };
    //!
    //! \brief 需要RECORD或者WAIT的Event
    //! \warning 支持先WAIT再RECORD以及先RECORD再WAIT时，需要使用aclrtCreateEventWithFlag接口创建Flag为ACL_EVENT_SYNC的Event
    //!
    aclrtEvent event;
    //!
    //! \brief OperatorType，支持UNDEFINED、RECORD和WAIT
    //!
    OperatorType operatorType = UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//!
//! \return bool值
//!
inline bool operator==(const EventParam &left, const EventParam &right)
{
    return left.operatorType == right.operatorType && left.event == right.event;
}

} // namespace common
} // namespace atb
#endif