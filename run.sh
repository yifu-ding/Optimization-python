# python3 main.py \
# 	--func_name "penalty" \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method "armijo" \
# 	--opt_method "lbfgs" \
# 	--max_iters 1e3 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m 1000 \
# 	--beta 1.1

python3 main.py \
	--func_name "trigonometric" \
	--stepsize_method "interpolate22" \
	--criterion_method "armijo" \
	--opt_method "lbfgs" \
	--max_iters 1e3 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m 1000 \
	--beta 1.1