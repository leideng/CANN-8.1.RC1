Function: This operation extracts a slice of size from a tensor input starting at the location specified by begin


## Slice算子介绍
### 1. 算子功能
从原始输入input数据中选择以begin开始的尺度大小为size的切片，其TensorFlow使用样例如下:
```
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
```
上述结果的来历：begin为[1,0,0]，size为[1,1,3]，为什么是三个数列表，因为原始数据维度为3。首先从起始位置开始[1,0,0]，第一位1，表示从axis=0的第二个位置开始，及从[[[3,3,3],[4,4,4]]]开始，尺度[1,1,3]的第一位为1，表示axis=0选择一个尺度的数据，即begin和size第一位共同作用，得到第一步数据[[[3,3,3],[4,4,4]]];

以上述结果为基础[[[3,3,3],[4,4,4]]]

第二位0，表示从axis=1的第1个位置开始，及从[[[3,3,3]]]开始，尺度[1,1,3]的第二位为1，表示axis=1选择一个尺度的数据，即begin和size第二位共同作用，得到第二步数据[[[3,3,3]]];

以上述结果为基础[[[3,3,3]]]

第三位0，表示从axis=2的第1个位置开始，及从[[[3]]]开始，尺度[1,1,3]的第三位为3，表示axis=2选择三个尺度的数据，即begin和size第三位共同作用，得到第二步数据[[[3,3,3]]]。

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
│       │   tensorflow_slice_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   slice.cc
│   │   slice.h
│   
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子实现目录
│           │   slice.py #算子动态实现
│       │   slice.py  #算子静态实现
│   └───op_info_cfg #算子信息库
│       └───aicore
│           └───ascend310
│               │   slice.ini 
└───testcase #测试用例
│   └───st
│       └───slice
│           └───ascend310
│               │   Slice_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───slice
│               │   test_slice_impl.py #算子实现ut
```
