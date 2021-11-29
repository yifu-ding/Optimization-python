# 最优化算法的 Python 程序实现

## 上机作业 1

### 编写程序

- 线搜索准则
	- 非精确线搜索的 Armijo 准则、Goldstein 准则、Wolfe 准则、强 Wolfe 准则
- Newton 法
	- 普通 Newton 法
	- 阻尼 Newton 方法
	- 混合 Newton 法
	- LM 方法
- 拟 Newton 法
	- 对称秩 1 方法 (SR1)
	- BFGS 方法
	- DFP 方法

### 实验题目

1) (16) Brown and Dennis function, m=4, 10, 20, 30, 40, 50. 

2) (22) Extended Powell singular function, m=20,40,60,80,100. 

3) (27) Brown almost linear function, n=20, 40, 60, 80, 100.


### 文件结构

```
|- run.sh
|- main.py
|- functions
	|- brown_almost_linear.py
	|- brown_and_dennis.py
	|- example.py 
	|- extended_powell_singular.py
|- methods
	|- criterion.py
	|- get_stepsize.py
	|- inexact.py
	|- newton.py
	|- quasi_newton.py
```

### 运行脚本示例

```sh
python3 main.py \
	--func_name name_of_objective_function \
	--stepsize_method method_of_getting_stepsize \
	--criterion_method method_of_criterion \
	--opt_method optimization_method \
	--max_iters maximum_iteration_numbers \
	--rho value_of_rho \
	--sigma value_of_sigma \
	--eps stopping_criterion \
	--m dimention_of_objective_function
```

更多参数详见主程序`./main.py#L20`
