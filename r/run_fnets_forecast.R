# One-step-ahead FNETS forecasts (optional forecasting module, Phase 7).
# Usage: Rscript r/run_fnets_forecast.R <manifest.csv> <params.csv>
# Each manifest input is a training window; the output _forecast.csv holds the 1-step-ahead
# forecast of every series on the original scale (the model is fitted on standardized data).
suppressMessages(library(fnets))
args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(script_dir, "common.R"))

fit_one <- function(x, params, prefix) {
  mu <- colMeans(x); s <- apply(x, 2, sd)
  z <- if (isTRUE(params$standardize)) scale(x) else x
  fn <- fnets(z, center = TRUE, fm.restricted = params$fm_restricted, q = params$q_method,
              ic.op = params$ic_op, var.order = params$var_order, var.method = params$var_method,
              do.threshold = params$do_threshold, do.lrpc = FALSE,
              tuning.args = list(tuning = params$tuning, n.folds = params$n_folds,
                                 path.length = params$path_length))
  fc <- predict(fn, n.ahead = 1, fc.restricted = params$fc_restricted)$forecast
  fc <- as.numeric(fc[1, ])
  if (isTRUE(params$standardize)) fc <- fc * s + mu
  write_mat(t(fc), paste0(prefix, "_forecast.csv"))
  write.csv(data.frame(q = fn$q), paste0(prefix, "_info.csv"), row.names = FALSE)
  TRUE
}

run_manifest(args[1], args[2], fit_one)
