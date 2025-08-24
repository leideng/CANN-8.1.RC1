Function: Applies a softmax function: log(exp(x) + 1)

## Softplus算子介绍
### 1. 算子功能
神经网络中常用的激活函数，计算表达式如下：
```
log(exp(x) + 1)
```


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Pytorch  |
| 实现方式 | DSL      |
| 动态/静态shape  | 动&静态 |

### 3. 主要工程结构
```
project
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   softplus.cc
│   │   softplus.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子动态实现
│           │   softplus.py
│       │   softplus.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   softplus.ini #算子信息库
│           └───ascend310p
│               │   softplus.ini #算子信息库
│           └───ascend910
│               │   softplus.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───softplus
│           └───ascend310
│               │   Softplus_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───softplus
│               │   test_softplus_impl.py #算子实现ut
```
