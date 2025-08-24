Function: Outputs a copy of the input tensor where values from the "height" and "width" dimensions are moved to the "depth" dimension.


## SpaceToDepth算子介绍
### 1. 算子功能
将数据从深度重新排列成空间数据块。

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
│       │   tensorflow_space_to_depth_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   space_to_depth.cc
│   │   space_to_depth.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子实现目录
│           │   space_to_depth.py #算子动态实现
│       │   space_to_depth.py  #算子静态实现
│   └───op_info_cfg #算子信息库
│       └───aicore
│           └───ascend310
│               │   space_to_depth.ini
│           └───ascend310p
│               │   space_to_depth.ini
│           └───ascend910
│               │   space_to_depth.ini
└───testcase #测试用例
│   └───st
│       └───space_to_depth
│           └───ascend310
│               │   SpaceToDepth_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───space_to_depth
│               │   test_space_to_depth_impl.py #算子实现ut
```
