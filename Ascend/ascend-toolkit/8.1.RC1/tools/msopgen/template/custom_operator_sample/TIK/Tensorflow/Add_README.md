Function: Adding two tensors: x1 + x2 = y


## Add算子介绍
### 1. 算子功能
Add算子主要实现两个tensor相加的功能，实现公式如下：
```
x1 + x2 = y
```


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
│       │   tensorflow_add_plugin.cc 
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   add.cc
│   │   add.h
│   
└───tbe
│   └───impl #算子实现目录
│       │   add.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   add.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───add
│           └───ascend310
│               │   Add_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───add
│               │   test_add_impl.py #算子实现ut
```
