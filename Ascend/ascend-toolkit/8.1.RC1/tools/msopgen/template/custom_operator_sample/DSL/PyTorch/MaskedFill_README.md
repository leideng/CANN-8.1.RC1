Function: Replace the value of X with a value according to mask.


## MaskedFill算子介绍
### 1. 算子功能
masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，
value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充。

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
│   │   masked_fill.cc
│   │   masked_fill.h
│
└───tbe
│   └───impl #算子实现目录
│       │   masked_fill.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   masked_fill.ini #算子信息库
│           └───ascend310p
│               │   masked_fill.ini #算子信息库
│           └───ascend910
│               │   masked_fill.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───masked_fill
│           └───ascend310
│               │   MaskedFill_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───masked_fill
│               │   test_masked_fill_impl.py #算子实现ut
```
