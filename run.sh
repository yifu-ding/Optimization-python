# python3 main.py \
# 	--func_name "penalty" \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method "armijo" \
# 	--opt_method "fr" \
# 	--max_iters 1e3 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m 1000 \
# 	--init_alpha 0.5

MEMO="extended_rosenbrock-interpolate22-strong_wolfe-lbfgs-alpha1.1"
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"

python3 main.py \
	--func_name "extended_rosenbrock" \
	--stepsize_method "interpolate22" \
	--criterion_method "strong_wolfe" \
	--opt_method "lbfgs" \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m 100 \
	--init_alpha 1.1 2>&1 | tee $log_filepath


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
