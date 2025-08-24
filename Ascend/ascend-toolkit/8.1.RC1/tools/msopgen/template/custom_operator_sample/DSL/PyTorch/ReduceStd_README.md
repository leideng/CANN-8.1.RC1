Function: Calculates the standard deviation and average value of Tensors.

## ReduceStd算子介绍
### 1. 算子功能
返回张量中所有元素的标准差和均值。

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
│   │   reduce_std.cc
│   │   reduce_std.h
│   
└───tbe
│   └───impl #算子实现目录
│       │   reduce_std.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   reduce_std.ini #算子信息库
│           └───ascend910
│               │   reduce_std.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───reduce_std
│           └───ascend310
│               │   ReduceStd_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───reduce_std
│               │   test_reduce_std_impl.py #算子实现ut
```
