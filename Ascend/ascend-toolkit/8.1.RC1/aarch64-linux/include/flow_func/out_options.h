/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description:
 */

#ifndef FLOW_FUNC_OUT_OPTIONS_H
#define FLOW_FUNC_OUT_OPTIONS_H

#include <memory>
#include "balance_config.h"

namespace FlowFunc {
class OutOptionsImpl;

class FLOW_FUNC_VISIBILITY OutOptions {
public:
    OutOptions();

    ~OutOptions();

    /**
     * @brief will get or create BalanceConfig
     * @return BalanceConfig
     */
    BalanceConfig *MutableBalanceConfig();

    /**
     * @brief get BalanceConfig
     * @return BalanceConfig null means not balance config
     */
    const BalanceConfig *GetBalanceConfig() const;

private:
    std::shared_ptr<OutOptionsImpl> impl_;
};
}
#endif // FLOW_FUNC_OUT_OPTIONS_H
