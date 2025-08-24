Function: Creates a criterion that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise.

## SmoothL1Loss算子介绍
### 1. 算子功能
按照分段函数规则计算两个Tensor间的损失误差


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
│   │   smooth_l1_loss.cc
│   │   smooth_l1_loss.h
│
└───tbe
│   └───impl #算子实现目录
│       │   smooth_l1_loss.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   smooth_l1_loss.ini #算子信息库
│           └───ascend310p
│               │   smooth_l1_loss.ini #算子信息库
│           └───ascend910
│               │   smooth_l1_loss.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───smooth_l1_loss
│           └───ascend310
│               │   SmoothL1Loss_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───smooth_l1_loss
│               │   test_smooth_l1_loss_impl.py #算子实现ut
```
