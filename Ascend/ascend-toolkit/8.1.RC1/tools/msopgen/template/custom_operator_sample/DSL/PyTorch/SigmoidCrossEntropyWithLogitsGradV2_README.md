Function: Compute gradients of cross entropy activated by sigmoid function in advanced.

## SigmoidCrossEntropyWithLogitsGradV2算子介绍
### 1. 算子功能
计算经sigmoid 函数激活之后的交叉熵的梯度。相比于SigmoidCrossEntropyWithLogitsGrad算子支持配置更多选项。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Pytorch  |
| 实现方式 | DSL      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   sigmoid_cross_entropy_with_logits_grad_v2.cc
│   │   sigmoid_cross_entropy_with_logits_grad_v2.h
│
└───tbe
│   └───impl #算子实现目录
│       │   sigmoid_cross_entropy_with_logits_grad_v2.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   sigmoid_cross_entropy_with_logits_grad_v2.ini #算子信息库
│           └───ascend310p
│               │   sigmoid_cross_entropy_with_logits_grad_v2.ini #算子信息库
│           └───ascend910
│               │   sigmoid_cross_entropy_with_logits_grad_v2.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───sigmoid_cross_entropy_with_logits_grad_v2
│           └───ascend310
│               │   SigmoidCrossEntropyWithLogitsGradV2_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───sigmoid_cross_entropy_with_logits_grad_v2
│               │   test_sigmoid_cross_entropy_with_logits_grad_v2_impl.py #算子实现ut
```
