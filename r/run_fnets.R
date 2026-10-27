# FNETS (Barigozzi, Cho and Owens, 2024) on one or more windows of returns.
# Usage: Rscript r/run_fnets.R <manifest.csv> <params.csv>
# Outputs per window (prefix from the manifest):
#   _beta.csv          idiosyncratic VAR coefficients as N x (N*p) [A_1 ... A_p], row i = equation i
#   _gamma.csv         idiosyncratic innovation covariance
#   _pcor.csv          idiosyncratic contemporaneous partial-correlation network
#   _lrpc.csv          idiosyncratic long-run partial-correlation network
#   _var_x.csv, _var_common.csv, _var_idio.csv   lag-0 variances of data, common and idiosyncratic parts
# Each window is standardized (mean 0, variance 1) when params$standardize is true.
#   _info.csv          number of factors q, factor criterion values, lambda, eta
suppressMessages(library(fnets))
args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(script_dir, "common.R"))

fit_one <- function(x, params, prefix) {
  if (isTRUE(params$standardize)) x <- scale(x)
  fn <- fnets(x, center = TRUE, fm.restricted = params$fm_restricted, q = params$q_method,
              ic.op = params$ic_op, var.order = params$var_order, var.method = params$var_method,
              do.threshold = params$do_threshold, do.lrpc = params$do_lrpc,
              tuning.args = list(tuning = params$tuning, n.folds = params$n_folds,
                                 path.length = params$path_length))
  p <- fn$idio.var$var.order
  beta <- fn$idio.var$beta                       # (N*p) x N, column j = equation of series j
  A <- t(beta)                                   # N x (N*p) = [A_1 ... A_p]
  write_mat(A, paste0(prefix, "_beta.csv"))
  write_mat(fn$idio.var$Gamma, paste0(prefix, "_gamma.csv"))
  write_mat(fn$lrpc$pc, paste0(prefix, "_pcor.csv"))
  write_mat(fn$lrpc$lrpc, paste0(prefix, "_lrpc.csv"))
  write_mat(diag(fn$acv$Gamma_x[, , 1]), paste0(prefix, "_var_x.csv"))
  write_mat(diag(fn$acv$Gamma_c[, , 1]), paste0(prefix, "_var_common.csv"))
  write_mat(diag(fn$acv$Gamma_i[, , 1]), paste0(prefix, "_var_idio.csv"))
  write.csv(data.frame(q = fn$q, q_method = params$q_method, ic_op = params$ic_op,
                       lambda = fn$idio.var$lambda, eta = fn$lrpc$eta, var_order = p),
            paste0(prefix, "_info.csv"), row.names = FALSE)
  TRUE
}

run_manifest(args[1], args[2], fit_one)
