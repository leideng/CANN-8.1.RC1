Function: A sparse algorithm to a single value or slice in the input data to obtain the output data.


## ScatterNdAdd算子介绍
### 1. 算子功能
ScatterNdAdd算子通过对输入数据中的单个值或切片应用稀疏算法，从而得到输出数据。

该算子具有var、indices和updates三个关键输入。其功能为使用updates更新var中indices指定位置的数据，即在var指定位置的数据上加上update的值。

三个输入之间的关系分别为：

-   张量var的shape的维度为P。
-   indices是整数张量，shape的维度（rank）为Q，索引为ref，最后一维的元素个数为K\(0<K<=P\)，shape为\[d\_0, ..., d\_\{Q-2\}, K\]。
-   张量updates的shape的维度（rank）为Q-1+P-K，shape为\[d\_0, ..., d\_\{Q-2\}, ref.shape\[K\], ..., ref.shape\[P-1\]\]。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Tensorflow  |
| 实现方式 | Tik      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_scatter_nd_add_plugin.cc # 算子插件文件
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   scatter_nd_add.cc
│   │   scatter_nd_add.h
│
└───tbe
│   └───impl #算子实现目录
│       │   scatter_nd_add.py
│   └───op_info_cfg #算子信息库
│       └───aicore
│           └───ascend310
│               │   scatter_nd_add.ini
│           └───ascend310p
│               │   scatter_nd_add.ini
│           └───ascend910
│               │   scatter_nd_add.ini
└───testcase #测试用例
│   └───st
│       └───scatter_nd_add
│           └───ascend310
│               │   ScatterNdAdd_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───ScatterNdAdd
│               │   test_scatter_nd_add_impl.py #算子实现ut
```
