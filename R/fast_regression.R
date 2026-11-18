# Fast embedding-regression internals.
#
# run_ols()        exact closed-form replacement for the estimatr::lm_robust(se_type = "stata") fit
#                  used by conText (weighted OLS, CR1 cluster-robust SEs for all D outcomes at once).
# run_jackknife()  closed-form leave-one-cluster-out jackknife. Each leave-out fit (coefficients and
#                  its own cluster-robust SEs) is obtained from per-cluster sufficient statistics instead
#                  of refitting lm_robust G times. Cost is linear in the number of clusters.
# ctm_permute_*()  permutation test with all permutations solved in batched matrix products.
# Random numbers are drawn with the same sample() calls, in the same order, as conText 3.0.1.

# model matrix with the same term names lm_robust/broom produce for `as.matrix(Y) ~ .`
ctm_design <- function(X) {
  Xdf <- as.data.frame(X)
  if (ncol(Xdf) == 0 || anyNA(Xdf)) return(NULL)
  mm <- stats::model.matrix(~ ., data = Xdf)
  if (nrow(mm) != nrow(Xdf)) return(NULL)
  if (qr(mm)$rank < ncol(mm)) return(NULL)  # rank-deficient: let the original code handle it
  storage.mode(mm) <- "double"
  mm
}

ctm_dense <- function(Y) { Y <- as.matrix(Y); storage.mode(Y) <- "double"; Y }

run_ols <- function(Y = NULL, X = NULL, ids = NULL) {
  if (ctm_original()) return(run_ols_orig(Y = Y, X = X, ids = ids))
  mm <- ctm_design(X)
  Ym <- ctm_dense(Y)
  if (is.null(mm) || anyNA(Ym)) return(run_ols_orig(Y = Y, X = X, ids = ids))
  ids <- factor(ids)
  weights <- 1/as.vector(table(ids)[ids])
  fit <- .Call(C_cluster_ols, mm, Ym, as.integer(ids) - 1L, nlevels(ids), as.double(weights), ctm_threads())
  terms <- colnames(mm)
  betas <- fit$B
  rownames(betas) <- terms
  keep <- terms != "(Intercept)"
  b2 <- rowSums(betas[keep, , drop = FALSE]^2)
  s2 <- rowSums(fit$se2[keep, , drop = FALSE])
  nm <- terms[keep]
  list('betas' = betas,
       'normed_betas_orig' = stats::setNames(b2, nm),
       'normed_betas_deflated' = stats::setNames(b2 - s2, nm),
       'beta_error_null' = stats::setNames(s2, nm))
}

ctm_jackknife_engine <- function(G) {
  e <- getOption("FastconText.jackknife", "auto")
  if (e == "cpu" || ctm_backend() == "cpu" || !metal_available()) return(0L)
  if (e == "metal" || ctm_backend() == "metal") return(1L)
  if (G >= getOption("FastconText.jackknife_gpu_min_clusters", 2000)) 1L else 0L
}

# G x K matrix of leave-one-cluster-out debiased squared norms (NULL -> caller falls back)
ctm_jackknife_partials <- function(X, Y, ids) {
  mm <- ctm_design(X)
  Ym <- ctm_dense(Y)
  if (is.null(mm) || anyNA(Ym) || ncol(mm) < 2) return(NULL)
  f <- factor(ids)
  weights <- 1/as.vector(table(f)[f])
  out <- .Call(C_jackknife, mm, Ym, as.integer(f) - 1L, nlevels(f), as.double(weights),
               ctm_threads(), ctm_jackknife_engine(nlevels(f)))
  colnames(out) <- colnames(mm)[-1]
  out
}

run_jackknife <- function(theta, X, Y, ids, confidence_level, verbose = F, parallel = F, jackknife_fraction = 1) {
  if (ctm_original()) return(run_jackknife_orig(theta = theta, X = X, Y = Y, ids = ids, confidence_level = confidence_level,
                                                 verbose = verbose, parallel = parallel, jackknife_fraction = jackknife_fraction))
  n <- nrow(X)
  if (jackknife_fraction < 1) {
    jackknife_sample <- sample(1:length(ids), size = length(ids)*jackknife_fraction, replace = F)
    ids <- ids[jackknife_sample]
    X <- as.data.frame(X[jackknife_sample, ])
    Y <- Y[jackknife_sample, ]
  }
  ids <- as.numeric(factor(ids))
  partials <- ctm_jackknife_partials(X, Y, ids)
  lost <- NULL
  if (is.null(partials)) {
    # exotic designs (NA, rank deficiency): original per-cluster refits on the same (sub)sample
    partials <- do.call(rbind, lapply(seq_len(max(ids)), function(i) jackknife_obs_remove(X, Y, ids, i)))
  } else {
    if (verbose) cat('jackknife:', nrow(partials), 'leave-one-out fits computed in closed form (', attr(partials, "backend"), ')\n')
    # leave-out fits that lost rank come back with NA in the dropped column
    bad <- which(apply(partials, 1, anyNA))
    if (length(bad)) {
      lost <- ctm_rank_lost(ctm_design(X), ids, bad)
      warning(lost$message, call. = FALSE)
      if (ctm_jackknife_rank() == "conText") {
        # refit those leave-outs with the original estimatr code, so the column it drops (chosen by
        # pivoted QR, which can be the intercept) and hence the result match conText exactly
        for (i in bad) { v <- jackknife_obs_remove(X, Y, ids, i); partials[i, ] <- v[colnames(partials)] }
      }
    }
  }
  out <- jackknife_calculate_se(partials, theta, n, confidence_level, jackknife_fraction)
  if (!is.null(lost) && ctm_jackknife_rank() == "na") {
    # coefficients outside the collinear set keep the same meaning in every leave-out fit; those
    # inside it do not, so their jackknife standard errors are not defined
    out[colnames(partials) %in% lost$terms, c("std.error", "lower.ci", "upper.ci")] <- NA_real_
  }
  out
}

ctm_jackknife_rank <- function() {
  r <- getOption("FastconText.jackknife_rank", "conText")
  if (!r %in% c("conText", "na"))
    stop('options(FastconText.jackknife_rank) must be "conText" or "na"', call. = FALSE)
  r
}

# Leave-out designs that lost rank: which coefficients take part in each collinearity, and how many
# contexts (and clusters) sit in the categories involved. Used for the warning and for the "na" mode.
ctm_rank_lost <- function(mm, ids, bad) {
  p <- ncol(mm)
  sets <- list()
  for (i in bad) {
    mi <- mm[ids != i, , drop = FALSE]
    r <- qr(mi)$rank
    if (r < p) {
      V <- svd(mi, nu = 0, nv = p)$v
      sets[[length(sets) + 1]] <- which(rowSums(abs(V[, (r + 1):p, drop = FALSE])) > 1e-6)
    }
  }
  sets <- unique(sets)
  clustered <- anyDuplicated(ids) > 0
  unit <- if (clustered) "clusters" else "contexts"
  count <- function(rows) {
    k <- sum(rows)
    if (clustered) sprintf("%d contexts in %d clusters", k, length(unique(ids[rows]))) else sprintf("%d %s", k, if (k == 1) "context" else "contexts")
  }
  describe <- function(cols) {
    terms <- setdiff(colnames(mm)[cols], "(Intercept)")
    dummies <- terms[vapply(terms, function(t) all(mm[, t] %in% c(0, 1)), logical(1))]
    sizes <- vapply(dummies, function(t) sprintf("%s %s", t, count(mm[, t] == 1)), character(1))
    # mutually exclusive dummies plus the intercept: the reference category is part of the problem
    if (1 %in% cols && length(dummies) > 1 && all(rowSums(mm[, dummies, drop = FALSE]) <= 1))
      sizes <- c(sprintf("reference category of %s %s", paste(dummies, collapse = "/"),
                         count(rowSums(mm[, dummies, drop = FALSE]) == 0)), sizes)
    paste0(paste(terms, collapse = ", "), if (length(sizes)) paste0(" (", paste(sizes, collapse = "; "), ")") else "")
  }
  terms <- unique(setdiff(colnames(mm)[unlist(sets)], "(Intercept)"))
  msg <- paste0("Jackknife: leaving out ", length(bad), " of ", max(ids), " ", unit,
                " makes the regression design rank-deficient, so the jackknife standard errors are not meaningful for: ",
                paste(vapply(sets, describe, character(1)), collapse = " | "), ". ",
                "This usually means a category has very few ", unit,
                "; merge sparse categories or make the most common level the reference category. ",
                "Results match conText by default; options(FastconText.jackknife_rank = \"na\") reports NA for these coefficients instead.")
  list(terms = terms, message = msg)
}

ctm_normed_rows <- function(B, vars) rowSums(B[vars, , drop = FALSE]^2)

# permutations without clusters: Y rows permuted, X fixed (same draws as permute_ols)
ctm_permute_plain <- function(X, E, num_permutations) {
  Xm <- cbind(as.matrix(X, ncol = ncol(X)), 1)
  colnames(Xm)[ncol(Xm)] <- "(Intercept)"
  storage.mode(Xm) <- "double"
  vars <- setdiff(colnames(X), "(Intercept)")
  E <- ctm_dense(E)
  n <- nrow(Xm); p <- ncol(Xm)
  A <- solve(crossprod(Xm))
  out <- matrix(NA_real_, num_permutations, length(vars), dimnames = list(NULL, vars))
  chunk <- max(1L, floor(getOption("FastconText.permute_chunk_entries", 2e8) / (p * n)))
  b <- 1L
  while (b <= num_permutations) {
    bb <- b:min(num_permutations, b + chunk - 1L)
    S <- matrix(0, length(bb) * p, n)
    for (t in seq_along(bb)) {
      idx <- sample(1:nrow(E))
      S[(t - 1L) * p + seq_len(p), ] <- t(Xm[order(idx), , drop = FALSE])
    }
    C <- ctm_gemm(S, E)
    for (t in seq_along(bb)) {
      Bt <- A %*% C[(t - 1L) * p + seq_len(p), , drop = FALSE]
      rownames(Bt) <- colnames(Xm)
      out[bb[t], ] <- ctm_normed_rows(Bt, vars)
    }
    b <- b + chunk
  }
  out
}

# permutations with clusters: replicates permute_ols(ids =) exactly, solving all draws in one batch
ctm_permute_cluster <- function(X, E, ids, num_permutations) {
  E <- ctm_dense(E)
  n <- nrow(E)
  vars <- setdiff(colnames(X), "(Intercept)")
  draws <- vector("list", num_permutations)
  for (b in seq_len(num_permutations)) {
    permuted_data <- dplyr::tibble(
      Y = matrix(seq_len(n), ncol = 1),
      X = X,
      id = ids
    ) %>%
      dplyr::group_by(id, X) %>%
      tidyr::nest(Y=Y) %>%
      dplyr::ungroup() %>%
      dplyr::mutate(
        Y = sample(Y)
      ) %>%
      tidyr::unnest(cols=c(Y))
    Xp <- permuted_data$X
    w <- 1/as.vector(table(permuted_data$id)[permuted_data$id])
    Xm <- cbind(as.matrix(Xp, ncol = ncol(Xp)), 1)
    colnames(Xm)[ncol(Xm)] <- "(Intercept)"
    storage.mode(Xm) <- "double"
    idx <- as.vector(permuted_data$Y[, 1])
    WX <- Xm * w
    draws[[b]] <- list(A = solve(crossprod(WX, Xm)), S = t(WX[order(idx), , drop = FALSE]), names = colnames(Xm))
  }
  p <- nrow(draws[[1]]$S)
  C <- ctm_gemm(do.call(rbind, lapply(draws, `[[`, "S")), E)
  out <- matrix(NA_real_, num_permutations, length(vars), dimnames = list(NULL, vars))
  for (b in seq_len(num_permutations)) {
    Bt <- draws[[b]]$A %*% C[(b - 1L) * p + seq_len(p), , drop = FALSE]
    rownames(Bt) <- draws[[b]]$names
    out[b, ] <- ctm_normed_rows(Bt, vars)
  }
  out
}
