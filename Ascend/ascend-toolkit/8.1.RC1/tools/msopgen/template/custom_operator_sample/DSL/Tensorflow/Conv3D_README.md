Function: Computes a 3D convolution given 5D "x" and "filter" tensors.

## Conv3D算子介绍
### 1. 算子功能
卷积层的常用计算单元，输入tensor通过卷积核进行卷积操作，得到输出tensor，主要处理5维tensor。


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
│       │   tensorflow_conv3d_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   conv3d.cc
│   │   conv3d.h
│
└───tbe
│   └───impl #算子实现目录
│       │   conv3d.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   conv3d.ini #算子信息库
│           └───ascend310P
│               │   conv3d.ini #算子信息库
│           └───ascend910
│               │   conv3d.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───conv3d
│           └───ascend310
│               │   Conv3D_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───conv3d
│               │   test_conv3d_impl.py #算子实现ut
```
