Function: Return x1 * x2 element-wise

## Mul算子介绍
### 1. 算子功能
计算两个tensor对应元素各自相乘


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
│       │   tensorflow_mul_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   mul.cc
│   │   mul.h
│
└───tbe
│   └───impl #算子实现目录
│       │   mul.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   mul.ini #算子信息库
│           └───ascend310p
│               │   mul.ini #算子信息库
│           └───ascend910
│               │   mul.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───mul
│           └───ascend310
│               │   Mul_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───mul
│               │   test_mul_impl.py #算子实现ut
```
