m=10000
alpha=0.5
criterion_method="strong_wolfe"
func_name="penalty"



# opt_method="cd"
# MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
# LOG_DIR="./logs"
# log_filepath=${LOG_DIR}/${MEMO}".log"
# python3 main.py \
# 	--func_name ${func_name} \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method ${criterion_method} \
# 	--opt_method ${opt_method} \
# 	--max_iters 1e4 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m ${m} \
# 	--init_alpha ${alpha} 2>&1 | tee $log_filepath

# opt_method="dy"
# MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
# LOG_DIR="./logs"
# log_filepath=${LOG_DIR}/${MEMO}".log"
# python3 main.py \
# 	--func_name ${func_name} \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method ${criterion_method} \
# 	--opt_method ${opt_method} \
# 	--max_iters 1e4 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m ${m} \
# 	--init_alpha ${alpha} 2>&1 | tee $log_filepath



# opt_method="bb"
# MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
# LOG_DIR="./logs"
# log_filepath=${LOG_DIR}/${MEMO}".log"
# python3 main.py \
# 	--func_name ${func_name} \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method ${criterion_method} \
# 	--opt_method ${opt_method} \
# 	--max_iters 1e4 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m ${m} \
# 	--init_alpha ${alpha} 2>&1 | tee $log_filepath


# opt_method="lbfgs"
# MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
# LOG_DIR="./logs"
# log_filepath=${LOG_DIR}/${MEMO}".log"
# python3 main.py \
# 	--func_name ${func_name} \
# 	--stepsize_method "interpolate22" \
# 	--criterion_method ${criterion_method} \
# 	--opt_method ${opt_method} \
# 	--max_iters 1e4 \
# 	--rho 1e-4 \
# 	--sigma 0.9 \
# 	--eps 1e-6 \
# 	--m ${m} \
# 	--init_alpha ${alpha} 2>&1 | tee $log_filepath



