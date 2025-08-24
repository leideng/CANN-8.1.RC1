Function: Sort the elements of the input tensor along a given dimension in ascending order by value.


## SortV2算子介绍
### 1. 算子功能
功能对标Pytorch官方类：torch.sort()，返回值不包含indices。

默认功能：将待排对象（input）中的元素，按照方向轴（axis）的方向进行升序排列，返回排序后的结果。


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
│   │   sort_v2.cc
│   │   sort_v2.h
│   
└───tbe
│   └───impl #算子实现目录
│       │   sort_v2.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend910
│               │   sort_v2.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───sort
│           └───ascend910
│               │   SortV2_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───sort_v2
│               │   test_sort_v2_impl.py #算子实现ut
```
