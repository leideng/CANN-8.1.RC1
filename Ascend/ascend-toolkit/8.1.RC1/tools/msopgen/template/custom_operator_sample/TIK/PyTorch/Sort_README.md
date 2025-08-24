Function: Sort the input tensor and return the value of index.


## Sort算子介绍
### 1. 算子功能
将待排对象（input）中的元素，按照方向轴（axis）的方向进行升序排列，返回排序后的结果以及各元素在input中的索引号。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Pytorch  |
| 实现方式 | Tik      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   sort.cc
│   │   sort.h
│   
└───tbe
│   └───impl #算子实现目录
│       │   sort.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   sort.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───sort
│           └───ascend310
│               │   Sort_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───sort
│               │   test_sort_impl.py #算子实现ut
```
