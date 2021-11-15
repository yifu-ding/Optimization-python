# 最优化算法的 Python 程序实现

## 上机作业 1

### 编写程序

- 线搜索
	- 精确线搜索
	- 非精确线搜索的 Armijo 搜索准则
- 最速下降法及 Newton 法
	- 阻尼 Newton 方法
	- 修正 Newton 法 (混合 Newton 法，LM 方法，稳定 Newton 法)
- 拟 Newton 法
	- 对称秩 1 方法 (SR1)
	- BFGS 方法
	- DFP 方法

### 数值实验

1) 比较不同Newton型方法的有效性, 通过实验, 分析方法的特点.

2) 对每个实验题目, 自选讨论一个问题.

3) 线搜索准则如选强 Wolfe 准则，可取ρ=10-4 , σ=0.9.

### 实验题目

见“More-testing”:

1) (16) Brown and Dennis function, m=4, 10, 20, 30, 40, 50. 

2) (22) Extended Powell singular function, m=20,40,60,80,100. 

3) (27) Brown almost linear function, n=20, 40, 60, 80, 100.