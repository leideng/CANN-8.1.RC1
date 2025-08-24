Function: Computes rectified linear: "max(x, 0)".

## Relu算子介绍
### 1. 算子功能
计算激活函数 relu，即 max(features, 0)。将大于0的保持不变，小于0的数置为0。

计算表达式如下：
```
f(x) = max(x,0)
```

### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | TensorFlow  |
| 实现方式 | DSL      |
| 动态/静态shape  | 动&静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_relu_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   relu.cc
│   │   relu.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子动态实现
│           │   relu.py
│       │   relu.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   relu.ini #算子信息库
│           └───ascend310p
│               │   relu.ini #算子信息库
│           └───ascend910
│               │   relu.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───relu
│           └───ascend310
│               │   Relu_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───relu
│               │   test_relu_impl.py #算子实现ut
```
