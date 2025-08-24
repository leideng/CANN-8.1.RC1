Function: Perform data normalization on FeatureMap

## BatchNorm算子介绍
### 1. 算子功能
Batchnorm是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法.


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
│       │   tensorflow_batch_norm_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   batch_norm.cc
│   │   batch_norm.h
│
└───tbe
│   └───impl #算子实现目录
│       │   batch_norm.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   batch_norm.ini #算子信息库
│           └───ascend310p
│               │   batch_norm.ini #算子信息库
│           └───ascend910
│               │   batch_norm.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───batch_norm
│           └───ascend310
│               │   BatchNorm_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───batch_norm
│               │   test_batch_norm_impl.py #算子实现ut
```
