python3 main.py \
	--func_name "trigonometric" \
	--stepsize_method "interpolate22" \
	--criterion_method "strong_wolfe" \
	--opt_method "bb" \
	--max_iters 1e5 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m 100 \
	--init_alpha 0.8

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
# 	--init_alpha 0.5


# python3 main.py \
# 	--func_name "penalty" \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method "strong_wolfe" \
# 	--opt_method "fr-prp" \
# 	--max_iters 1e3 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m 10 \
# 	--init_alpha 0.8
