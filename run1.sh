m=100
opt_method="lbfgs"
criterion_method="armijo"
func_name="penalty"

alpha=3
MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"
python3 main.py \
	--func_name ${func_name} \
	--stepsize_method "interpolate22" \
	--criterion_method ${criterion_method} \
	--opt_method ${opt_method} \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m ${m} \
	--init_alpha ${alpha} 2>&1 | tee $log_filepath



alpha=4
MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"
python3 main.py \
	--func_name ${func_name} \
	--stepsize_method "interpolate22" \
	--criterion_method ${criterion_method} \
	--opt_method ${opt_method} \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m ${m} \
	--init_alpha ${alpha} 2>&1 | tee $log_filepath

alpha=5
MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"
python3 main.py \
	--func_name ${func_name} \
	--stepsize_method "interpolate22" \
	--criterion_method ${criterion_method} \
	--opt_method ${opt_method} \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m ${m} \
	--init_alpha ${alpha} 2>&1 | tee $log_filepath

alpha=6
MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"
python3 main.py \
	--func_name ${func_name} \
	--stepsize_method "interpolate22" \
	--criterion_method ${criterion_method} \
	--opt_method ${opt_method} \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m ${m} \
	--init_alpha ${alpha} 2>&1 | tee $log_filepath


alpha=7
MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"
python3 main.py \
	--func_name ${func_name} \
	--stepsize_method "interpolate22" \
	--criterion_method ${criterion_method} \
	--opt_method ${opt_method} \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m ${m} \
	--init_alpha ${alpha} 2>&1 | tee $log_filepath


alpha=8
MEMO="${func_name}-${criterion_method}-"${opt_method}"-m"${m}"-alpha"${alpha}
LOG_DIR="./logs"
log_filepath=${LOG_DIR}/${MEMO}".log"
python3 main.py \
	--func_name ${func_name} \
	--stepsize_method "interpolate22" \
	--criterion_method ${criterion_method} \
	--opt_method ${opt_method} \
	--max_iters 1e4 \
	--rho 1e-4 \
	--sigma 0.9 \
	--eps 1e-6 \
	--m ${m} \
	--init_alpha ${alpha} 2>&1 | tee $log_filepath