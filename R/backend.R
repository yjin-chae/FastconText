#' Metal backend status and control
#'
#' `metal_info()` reports whether an Apple GPU is usable from this R process.
#' `metal_available()` returns `TRUE`/`FALSE`.
#'
#' The computational backend is chosen with `options(FastconText.backend = ...)`:
#' \describe{
#'   \item{`"auto"` (default)}{use the GPU for large float32 workloads (similarities against the
#'   vocabulary, bootstrap and permutation batches, large document-embedding products) when a Metal
#'   device is available; everything else runs in exact double precision on the CPU.}
#'   \item{`"metal"`}{force every accelerated operation onto the GPU regardless of size.}
#'   \item{`"cpu"`}{never use the GPU; still uses the fast closed-form algorithms (double precision).}
#'   \item{`"original"`}{run the original conText 3.0.1 code paths (useful for validation).}
#' }
#' Further options: `FastconText.gpu_min_flops` (size threshold for `"auto"`, default `1e8`),
#' `FastconText.jackknife` (`"auto"`, `"metal"` or `"cpu"`; engine for the per-cluster jackknife loop),
#' and `FastconText.threads` (CPU threads, default: all physical cores).
#' `FastconText.jackknife_rank` (`"conText"` (default) or `"na"`) controls jackknife leave-out fits that
#' become rank-deficient: refit them with the original code so results match conText, or report `NA`
#' standard errors and confidence intervals for the coefficients involved. Both settings warn.
#' `FastconText.na_action` (`"conText"` (default) or `"drop"`) controls contexts with missing covariate
#' values in `conText()`: keep them as conText 3.0.1 does, or drop them before the regression (listwise,
#' like `lm()`'s `na.omit`), with a message giving the number dropped.
#'
#' @return `metal_info()`: a list with `available`, `device`, `max_buffer_bytes`, `reason`.
#' @export
#' @rdname metal_info
metal_info <- function() {
  if (is.null(.ctm$info)) .ctm$info <- .Call(C_metal_info)
  .ctm$info
}

#' @export
#' @rdname metal_info
metal_available <- function() isTRUE(metal_info()$available)

.ctm <- new.env(parent = emptyenv())

ctm_backend <- function() {
  b <- getOption("FastconText.backend", "auto")
  if (!b %in% c("auto", "metal", "cpu", "original"))
    stop('options(FastconText.backend) must be one of "auto", "metal", "cpu", "original"', call. = FALSE)
  b
}
ctm_original <- function() identical(ctm_backend(), "original")
ctm_threads <- function() {
  t <- getOption("FastconText.threads", NA)
  if (is.na(t)) t <- parallel::detectCores(logical = FALSE)
  if (identical(Sys.getenv("_R_CHECK_LIMIT_CORES_"), "TRUE")) t <- min(t, 2L)
  if (is.na(t) || t < 1) t <- 1L
  as.integer(t)
}
# should an operation costing ~`flops` run on the GPU?
ctm_gpu <- function(flops) {
  b <- ctm_backend()
  if (b %in% c("cpu", "original")) return(FALSE)
  if (!metal_available()) return(FALSE)
  b == "metal" || flops >= getOption("FastconText.gpu_min_flops", 1e8)
}

# evaluate expr with GPU float32 disabled (precision-critical inputs to regressions / solves)
ctm_double <- function(expr) {
  if (ctm_original()) return(expr)
  old <- options(FastconText.backend = "cpu")
  on.exit(options(old))
  expr
}

# ------------------------------------------------------------------ linear-algebra wrappers

# op(A) %*% op(B)
ctm_gemm <- function(A, B, ta = FALSE, tb = FALSE) {
  A <- as.matrix(A); B <- as.matrix(B)
  storage.mode(A) <- "double"; storage.mode(B) <- "double"
  m <- if (ta) ncol(A) else nrow(A); k <- if (ta) nrow(A) else ncol(A); n <- if (tb) nrow(B) else ncol(B)
  if (ctm_gpu(2 * m * k * n)) return(.Call(C_metal_gemm, A, B, ta, tb))
  if (ta) A <- t(A)
  if (tb) B <- t(B)
  A %*% B
}

# diag(scale) %*% M %*% E for a sparse M (N x V) and dense E (V x D); returns a base matrix
ctm_spmm <- function(M, E, scale) {
  E <- as.matrix(E); storage.mode(E) <- "double"
  M <- methods::as(methods::as(M, "CsparseMatrix"), "generalMatrix")
  if (ctm_gpu(2 * length(M@x) * ncol(E))) {
    Mr <- methods::as(M, "RsparseMatrix")
    out <- .Call(C_metal_spmm, as.integer(Mr@p), as.integer(Mr@j), as.double(Mr@x), E, as.double(scale))
  } else {
    out <- as.matrix(M %*% E) * scale
  }
  dimnames(out) <- list(rownames(M), colnames(E))
  out
}

# cosine similarity matrix (text2vec::sim2(x, y, method = "cosine", norm = "l2") semantics)
ctm_sim2 <- function(x, y) {
  x <- as.matrix(x); y <- as.matrix(y)
  if (ctm_gpu(2 * nrow(x) * nrow(y) * ncol(x))) {
    storage.mode(x) <- "double"; storage.mode(y) <- "double"
    if (nrow(x) >= nrow(y)) s <- t(.Call(C_metal_sim_reduce, x, y, 1L, nrow(y), FALSE, 2L, 0L, 0L, 0)[[1]])
    else s <- .Call(C_metal_sim_reduce, y, x, 1L, nrow(x), FALSE, 2L, 0L, 0L, 0)[[1]]
    dimnames(s) <- list(rownames(x), rownames(y))
    return(s)
  }
  text2vec::sim2(x, y, method = "cosine", norm = "l2")
}

# Batched similarities between the queries Q (rows ordered b*ng + g) and the rows of V,
# reduced over the nb draws for each group g (and for the ratio g1/g2 when ratio = TRUE).
#  op = "stats": list(mean, sd, lo, hi), each nq x nrow(V); lo/hi are the klo-th/khi-th smallest draws
#  op = "count": nq x nrow(V) counts of draws >= thr (groups) and |ratio - 1| >= thr (ratio)
ctm_sim_reduce <- function(V, Q, nb, ng, ratio = FALSE, op = c("stats", "count"), klo = 1L, khi = 1L, thr = 0) {
  op <- match.arg(op)
  V <- as.matrix(V); Q <- as.matrix(Q)
  storage.mode(V) <- "double"; storage.mode(Q) <- "double"
  opi <- if (op == "stats") 0L else 1L
  if (op == "count") { thr <- as.double(thr); stopifnot(length(thr) == (ng + ratio) * nrow(V)) }
  if (ctm_gpu(2 * nrow(Q) * nrow(V) * ncol(V))) {
    out <- .Call(C_metal_sim_reduce, V, Q, as.integer(nb), as.integer(ng), ratio, opi, as.integer(klo), as.integer(khi), thr)
  } else {
    out <- .Call(C_sim_reduce_cpu, V, Q, as.integer(nb), as.integer(ng), ratio, opi, as.integer(klo), as.integer(khi), thr, ctm_threads())
  }
  if (op == "stats") names(out) <- c("mean", "sd", "lo", "hi") else out <- out[[1]]
  out
}

# options(FastconText.na_action): "conText" keeps contexts with missing covariates (conText 3.0.1
# behavior); "drop" removes them before the regression, jackknife and permutations.
ctm_na_action <- function() {
  a <- getOption("FastconText.na_action", "conText")
  if (!(length(a) == 1 && a %in% c("conText", "drop")))
    stop('options(FastconText.na_action) must be "conText" or "drop"', call. = FALSE)
  a
}
