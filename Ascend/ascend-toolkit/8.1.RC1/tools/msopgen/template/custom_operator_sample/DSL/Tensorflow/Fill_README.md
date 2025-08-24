Function: Creates a tensor filled with a scalar value.

## Fill算子介绍
### 1. 算子功能
生成一个所有元素都是同一标量的tensor。其TensorFlow框架下的使用方式如下：
```
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```


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
│       │   tensorflow_fill_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   fill.cc
│   │   fill.h
│   
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子动态实现
│           │   fill.py
│       │   fill.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   fill.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───fill
│           └───ascend310
│               │   Fill_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───fill
│               │   test_fill_impl.py #算子实现ut
```
