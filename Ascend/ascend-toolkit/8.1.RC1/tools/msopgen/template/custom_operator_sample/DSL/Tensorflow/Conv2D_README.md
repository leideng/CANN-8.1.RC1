Function: Computes a 2D convolution given 4D "x" and "filter" tensors.

## Conv2D算子介绍
### 1. 算子功能
卷积层的常用计算单元，输入tensor通过卷积核进行卷积操作，得到输出tensor，主要处理4维tensor。



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
│       │   tensorflow_conv2d_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   conv2d.cc
│   │   conv2d.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子动态实现
│           │   conv2d.py
│       │   conv2d.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   conv2d.ini #算子信息库
│           └───ascend310P
│               │   conv2d.ini #算子信息库
│           └───ascend910
│               │   conv2d.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───conv2d
│           └───ascend310
│               │   Conv2D_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───conv2d
│               │   test_conv2d_impl.py #算子实现ut
```
