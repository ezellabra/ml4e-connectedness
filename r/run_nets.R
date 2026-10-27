# NETS (Barigozzi and Brownlees, 2019) on one or more windows of returns.
# Usage: Rscript r/run_nets.R <manifest.csv> <params.csv>
# Outputs per window (prefix from the manifest):
#   _A.csv      N x (N*p) autoregressive coefficients [A_1 ... A_p], row i = equation of series i
#   _sigma.csv  innovation covariance (inverse of the estimated concentration matrix)
#   _pcor.csv   contemporaneous partial-correlation network
#   _lrpc.csv   long-run partial-correlation network
#   _info.csv   selected lambda, BIC, number of non-zero parameters
# Each window is demeaned, and standardized when params$standardize is true.
suppressMessages(library(nets))
args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(script_dir, "common.R"))

fit_nets <- function(y, p, lambda, params) {
  invisible(capture.output(
    m <- nets(y, GN = TRUE, CN = TRUE, p = p, lambda = lambda,
              algorithm = params$algorithm, weights = params$weights,
              iter.in = params$iter_in, iter.out = params$iter_out, verbose = FALSE)))
  m
}

bic_nets <- function(m, y, p) {
  N <- ncol(y); Tn <- nrow(y)
  E <- y[(p + 1):Tn, , drop = FALSE]
  for (k in 1:p) E <- E - y[(p + 1 - k):(Tn - k), , drop = FALSE] %*% t(m$A.hat[, , k])
  S <- crossprod(E) / nrow(E)
  C <- m$C.hat
  ld <- determinant(C, logarithm = TRUE)
  if (ld$sign <= 0) return(Inf)
  n_par <- sum(m$A.hat != 0) + sum(C[upper.tri(C)] != 0)
  nrow(E) * (-as.numeric(ld$modulus) + sum(diag(C %*% S))) + log(nrow(E)) * n_par
}

fit_one <- function(x, params, prefix) {
  y <- scale(x, center = TRUE, scale = isTRUE(params$standardize))
  p <- params$lags
  grid <- exp(seq(log(params$lambda_max), log(params$lambda_max * params$lambda_min_ratio),
                  length.out = params$lambda_grid_size))
  best <- NULL; best_bic <- Inf; best_lambda <- NA
  for (lam in grid) {
    m <- tryCatch(fit_nets(y, p, lam, params), error = function(e) NULL)
    if (is.null(m)) next
    b <- bic_nets(m, y, p)
    if (b < best_bic) { best <- m; best_bic <- b; best_lambda <- lam }
  }
  if (is.null(best)) stop("NETS failed for every lambda in the grid")
  N <- ncol(y)
  A <- do.call(cbind, lapply(1:p, function(k) best$A.hat[, , k]))
  C <- best$C.hat
  A1 <- diag(N)
  for (k in 1:p) A1 <- A1 - best$A.hat[, , k]
  K_lr <- t(A1) %*% C %*% A1
  write_mat(A, paste0(prefix, "_A.csv"))
  write_mat(solve(C), paste0(prefix, "_sigma.csv"))
  write_mat(partial_cor(C), paste0(prefix, "_pcor.csv"))
  write_mat(partial_cor(K_lr), paste0(prefix, "_lrpc.csv"))
  write.csv(data.frame(lambda = best_lambda, bic = best_bic,
                       n_granger = sum(best$A.hat != 0), n_contemp = sum(C[upper.tri(C)] != 0)),
            paste0(prefix, "_info.csv"), row.names = FALSE)
  TRUE
}

run_manifest(args[1], args[2], fit_one)
