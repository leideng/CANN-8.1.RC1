Function: Calculate data's square.


## Square算子介绍
### 1. 算子功能
计算tensor中各元素的平方。计算公式如下：
```
y = x * x
```


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
│       │   square_impl.py
└───proto #算子原型目录
│   │   square.py
└───testcase #测试用例
│   └───st
│       └───square
│           │   Square_case_sample.json #算子st用例
│   └───ut
│       └───ops_test
│           │   test_square_impl.py #算子实现ut
```
