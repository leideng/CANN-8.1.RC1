Function: Compute norm for p equals 0, 1, 2, -inf, inf, or other integers.

## LpNorm算子介绍
### 1. 算子功能
计算输入input的范数。 norm范数支持0、1、inf、-inf、其他等情况：

计算过程：p默认为2

* p = 0: 矩阵或向量中的非0元素的个数；
* p = 1: 矩阵或向量中的每个元素绝对值abs之和；
* p = inf: 矩阵或向量中各项元素绝对值abs中的最大值；
* p = -inf: 矩阵或向量中各项元素绝对值abs中的最小值；(inf与-inf芯片不支持，不要实现)
* p =其他（包括2）: 矩阵或向量中各项元素绝对值abs的pow(p，即p次幂)之和再开p次方.


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Pytorch  |
| 实现方式 | DSL      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   lp_norm.cc
│   │   lp_norm.h
│
└───tbe
│   └───impl #算子实现目录
│       │   lp_norm.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   lp_norm.ini #算子信息库
│           └───ascend310p
│               │   lp_norm.ini #算子信息库
│           └───ascend910
│               │   lp_norm.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───lp_norm
│           └───ascend310
│               │   LpNorm_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───lp_norm
│               │   test_lp_norm_impl.py #算子实现ut
```
