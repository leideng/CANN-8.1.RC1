/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Description: Huawei Code
 *
 * Author: Huawei
 *
 */
#include "add_dsl.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(AddDSLInferShape)
{
    TensorDesc tensordesc_output = op.GetOutputDescByName("y");

    tensordesc_output.SetShape(op.GetInputDescByName("x1").GetShape());
    tensordesc_output.SetDataType(op.GetInputDescByName("x1").GetDataType());
    tensordesc_output.SetFormat(op.GetInputDescByName("x1").GetFormat());

    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AddDSL, AddDSLVerify)
{
    if (op.GetInputDescByName("x1").GetDataType() != op.GetInputDescByName("x2").GetDataType()) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AddDSL, AddDSLInferShape);

VERIFY_FUNC_REG(AddDSL, AddDSLVerify);
}