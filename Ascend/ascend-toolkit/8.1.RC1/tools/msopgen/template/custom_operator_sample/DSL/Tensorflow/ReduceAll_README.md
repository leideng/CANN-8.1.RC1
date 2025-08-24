Function: Calculate the "logical sum" of elements of a tensor in a dimension.

## ReduceAll算子介绍
### 1. 算子功能
计算一个张量在维度上元素的“逻辑和”.
给按照轴线给定的维度减少input_tensor .除非keep_dims为 true,否则张量的秩将在轴的每个条目中减少1.如果keep_dims为 true,则减小的维度将保留为长度1.

如果轴没有条目,则会减少所有维度,并返回具有单个元素的张量.

TensorFlow使用样例如下：
```
x = tf.constant([[True,  True], [False, False]])
tf.reduce_all(x)  # False
tf.reduce_all(x, 0)  # [False, False]
tf.reduce_all(x, 1)  # [True, False]
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
│       │   tensorflow_reduce_all_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   reduce_all.cc
│   │   reduce_all.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子动态实现
│           │   reduce_all.py
│       │   reduce_all.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   reduce_all.ini #算子信息库
│           └───ascend310p
│               │   reduce_all.ini #算子信息库
│           └───ascend910
│               │   reduce_all.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───reduce_all
│           └───ascend310
│               │   ReduceAll_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───reduce_all
│               │   test_reduce_all_impl.py #算子实现ut
```
