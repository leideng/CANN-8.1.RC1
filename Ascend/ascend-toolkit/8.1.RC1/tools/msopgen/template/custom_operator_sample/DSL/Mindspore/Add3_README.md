Function: Add all input tensors element-wise.


## Add3算子介绍
### 1. 算子功能
把两个tensor所有元素相加。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Mindspore  |
| 实现方式 | DSL      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───mindspore
│   └───impl #原型公共依赖
│       │   add3_impl.py
└───proto #算子原型目录
│   │   add3.py
└───testcase #测试用例
│   └───st
│       └───add3
│           │   Add3_case_sample.json #算子st用例
│   └───ut
│       └───ops_test
│           │   test_add3_impl.py #算子实现ut
```
