Function: Pads a tensor.


## Pad算子介绍
### 1. 算子功能
将目标张量按照具体padding要求进行填充，其TensorFlow使用样例如下:
```
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
tf.pad(t, paddings)

# [[0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 1, 2, 3, 0, 0],
#  [0, 0, 4, 5, 6, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0]]
```


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Tensorflow  |
| 实现方式 | Tik      |
| 动态/静态shape  | 动态&静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_pad_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   pad.cc
│   │   pad.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子实现目录
│           │   pad.py #算子动态实现
│       │   pad.py  #算子静态实现
│   └───op_info_cfg #算子信息库
│       └───aicore
│           └───ascend310
│               │   pad.ini
│           └───ascend310p
│               │   pad.ini
│           └───ascend910
│               │   pad.ini
└───testcase #测试用例
│   └───st
│       └───pad
│           └───ascend310
│               │   Pad_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───pad
│               │   test_pad_impl.py #算子实现ut
```
