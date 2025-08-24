Function: Perform max pooling on the input.

## MaxPool算子介绍
### 1. 算子功能
卷积神经网络池化层中的常用计算单元，主要是计算某一区域内的最大值。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | TensorFlow  |
| 实现方式 | DSL      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_max_pool_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   max_pool.cc
│   │   max_pool.h
│
└───tbe
│   └───impl #算子实现目录
│       │   max_pool.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   max_pool.ini #算子信息库
│           └───ascend310p
│               │   max_pool.ini #算子信息库
│           └───ascend910
│               │   max_pool.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───max_pool
│           └───ascend310
│               │   MaxPool_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───max_pool
│               │   test_max_pool_impl.py #算子实现ut
```
