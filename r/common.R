# Shared helpers for the R scripts called from src/r_bridge.py.
# Usage pattern: Rscript r/run_<method>.R <manifest.csv> <params.csv>
#   manifest.csv: columns id, input, output_prefix (one row per window; input is a CSV with a
#                 date column followed by one column per series)
#   params.csv:   columns key, value (written by src/r_bridge.py from config.yaml)

read_params <- function(path) {
  p <- read.csv(path, stringsAsFactors = FALSE)
  out <- as.list(p$value)
  names(out) <- p$key
  num <- suppressWarnings(as.numeric(out))
  for (k in seq_along(out)) if (!is.na(num[k])) out[[k]] <- num[k]
  for (k in seq_along(out)) if (identical(out[[k]], "true")) out[[k]] <- TRUE
  for (k in seq_along(out)) if (identical(out[[k]], "false")) out[[k]] <- FALSE
  out
}

read_window <- function(path) {
  d <- read.csv(path, check.names = FALSE)
  x <- as.matrix(d[, -1, drop = FALSE])
  colnames(x) <- colnames(d)[-1]
  x
}

write_mat <- function(m, path) {
  write.csv(m, path, row.names = FALSE)
}

# Partial correlations from a concentration (precision) matrix.
partial_cor <- function(K) {
  d <- sqrt(abs(diag(K)))
  pc <- -K / outer(d, d)
  diag(pc) <- 1
  pc
}

run_manifest <- function(manifest_path, params_path, fit_one) {
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  params <- read_params(params_path)
  n_fail <- 0
  for (r in seq_len(nrow(manifest))) {
    set.seed(params$seed + manifest$id[r])
    x <- read_window(manifest$input[r])
    res <- tryCatch(fit_one(x, params, manifest$output_prefix[r]),
                    error = function(e) {
                      writeLines(conditionMessage(e), paste0(manifest$output_prefix[r], "_error.txt"))
                      NULL
                    })
    if (is.null(res)) n_fail <- n_fail + 1
  }
  cat(sprintf("Processed %d windows, %d failures\n", nrow(manifest), n_fail))
}
