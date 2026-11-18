# Fast bootstrap / permutation internals.
#
# The original functions resample the document-embedding matrix, recompute group means and then
# compare them with the full vocabulary once per draw, collecting B x G x V rows in a data frame
# before summarising. Here all draws are expressed as one sparse "averaging" matrix, the B x G
# mean embeddings come from one sparse x dense product, and similarities plus per-column reductions
# (mean, sd, order statistics, exceedance counts) run in a single batched pass (Metal GPU when
# available, multi-threaded C++ otherwise). Index draws use the original sample() calls in the
# original order, so set.seed() reproduces conText 3.0.1 results.

# index vector drawn exactly as dem_sample(x, size, replace, by) draws it
ctm_dem_sample_idx <- function(x, size = 1, replace = TRUE, by = NULL) {
  if (!is.null(by)) {
    if ((length(by) == 1 && !(by %in% colnames(x@docvars))) || (length(by) != 1 && length(by) != nrow(x)))
      stop("by must either be a character vector of length 1, referring to a variable present in @docvars, or a vector of length equal to nrow(x)", call. = FALSE)
    if (length(by) == nrow(x)) {
      dv <- x@docvars
      dv["tmp_by"] <- by
      dv %>% dplyr::mutate(docid = seq_len(nrow(.))) %>% dplyr::group_by(tmp_by) %>%
        dplyr::sample_frac(size = size, replace = replace) %>% dplyr::pull(docid)
    } else {
      x@docvars %>% dplyr::mutate(docid = seq_len(nrow(.))) %>% dplyr::group_by_at(by) %>%
        dplyr::sample_frac(size = size, replace = replace) %>% dplyr::pull(docid)
    }
  } else {
    data.frame(docid = seq_len(nrow(x))) %>% dplyr::sample_frac(size = size, replace = replace) %>% dplyr::pull(docid)
  }
}

# groups usable by the fast path (mirrors dem_group's rowsum/table alignment)
ctm_group_levels <- function(groups) {
  if (is.null(groups) || anyNA(groups)) return(NULL)
  if (is.factor(groups) && any(table(groups) == 0)) return(NULL)
  sort(unique(groups))
}

# (B*G) x D matrix of group means; row (b-1)*G + g averages E[idx_b, ] over rows with label levels[g]
ctm_group_means_batch <- function(E, idx_list, labels_list, levels) {
  G <- length(levels); B <- length(idx_list)
  ii <- vector("list", B); jj <- vector("list", B); xx <- vector("list", B)
  for (b in seq_len(B)) {
    idx <- idx_list[[b]]
    gi <- match(as.character(labels_list[[b]]), as.character(levels))
    cnt <- tabulate(gi, G)
    ii[[b]] <- (b - 1L) * G + gi; jj[[b]] <- idx; xx[[b]] <- 1 / cnt[gi]
  }
  W <- Matrix::sparseMatrix(i = unlist(ii), j = unlist(jj), x = unlist(xx), dims = c(B * G, nrow(E)))
  ctm_spmm(W, E, rep(1, B * G))
}

ctm_kidx <- function(confidence_level, nb) c(round((1 - confidence_level) * nb), round(confidence_level * nb))

# C-locale ordering used by dplyr::group_by() for character keys
ctm_order <- function(...) order(..., method = "radix")

# ------------------------------------------------------------------ get_nns (bootstrap = TRUE)
ctm_boot_nns <- function(x_dem, N, candidates, pre_trained, num_bootstraps, confidence_level) {
  groups <- x_dem@docvars$group
  lev <- ctm_group_levels(groups)
  if (is.null(lev)) return(NULL)
  glab <- as.character(lev); G <- length(lev)
  idx_list <- replicate(num_bootstraps, ctm_dem_sample_idx(x_dem, size = 1, replace = TRUE, by = groups), simplify = FALSE)
  Q <- ctm_group_means_batch(as.matrix(x_dem), idx_list, lapply(idx_list, function(i) groups[i]), lev)
  V <- if (length(candidates) > 0) pre_trained[rownames(pre_trained) %in% candidates, , drop = FALSE] else pre_trained
  k <- ctm_kidx(confidence_level, num_bootstraps)
  st <- ctm_sim_reduce(V, Q, nb = num_bootstraps, ng = G, op = "stats", klo = k[1], khi = k[2])
  feats <- rownames(V)
  res <- dplyr::tibble(target = factor(rep(glab, times = length(feats)), levels = glab),
                       feature = rep(feats, each = G),
                       std.error = c(st$sd), value = c(st$mean), lower.ci = c(st$lo), upper.ci = c(st$hi))
  res <- res[ctm_order(as.integer(res$target), res$feature), ]
  res %>%
    dplyr::group_by(target) %>%
    dplyr::slice_max(order_by = value, n = N) %>%
    dplyr::mutate(rank = 1:dplyr::n()) %>%
    dplyr::ungroup() %>%
    dplyr::select('target', 'feature', 'rank', 'value', 'std.error', 'lower.ci', 'upper.ci')
}

# ------------------------------------------------------------------ get_cos_sim (bootstrap = TRUE)
ctm_boot_cos_sim <- function(x_dem, features, pre_trained, num_bootstraps, stem, language) {
  groups <- x_dem@docvars$group
  lev <- ctm_group_levels(groups)
  if (is.null(lev)) return(NULL)
  glab <- as.character(lev); G <- length(lev)
  idx_list <- replicate(num_bootstraps, ctm_dem_sample_idx(x_dem, size = 1, replace = TRUE, by = groups), simplify = FALSE)
  Q <- ctm_group_means_batch(as.matrix(x_dem), idx_list, lapply(idx_list, function(i) groups[i]), lev)
  lapply(seq_len(num_bootstraps), function(b) {
    wvs <- Q[(b - 1L) * G + seq_len(G), , drop = FALSE]
    rownames(wvs) <- glab
    cos_sim(x = wvs, pre_trained = pre_trained, features = features, stem = stem, language = language, as_list = FALSE, show_language = FALSE)
  })
}

# ------------------------------------------------------------------ get_ncs (bootstrap = TRUE)
ctm_boot_ncs <- function(x_dem, x_contexts, by, N, num_bootstraps, confidence_level) {
  groups <- x_dem@docvars$group
  lev <- ctm_group_levels(groups)
  if (is.null(lev) || is.null(by) || length(by) != nrow(x_dem)) return(NULL)
  dids <- as.character(quanteda::docid(x_contexts))
  if (anyDuplicated(dids) || anyDuplicated(rownames(x_dem)) || !all(rownames(x_dem) %in% dids)) return(NULL)
  glab <- as.character(lev); G <- length(lev)
  E <- as.matrix(x_dem)
  ctx_text <- sapply(x_contexts, function(i) paste(i, collapse = " "))[match(rownames(x_dem), dids)]
  # identical context strings are merged by the original group_by(target, context)
  rep_row <- !duplicated(ctx_text)
  mult <- as.vector(table(ctx_text)[ctx_text[rep_row]])
  if (any(mult > 1)) {
    grp <- match(ctx_text, ctx_text[rep_row])
    for (u in which(mult > 1)) {
      rows <- which(grp == u)
      if (max(abs(sweep(E[rows, , drop = FALSE], 2, E[rows[1], ]))) > 0) return(NULL)
    }
  }
  idx_list <- replicate(num_bootstraps, ctm_dem_sample_idx(x_dem, size = 1, replace = TRUE, by = by), simplify = FALSE)
  Q <- ctm_group_means_batch(E, idx_list, lapply(idx_list, function(i) groups[i]), lev)
  k <- ctm_kidx(confidence_level, num_bootstraps)
  B <- num_bootstraps
  st <- ctm_sim_reduce(E[rep_row, , drop = FALSE], Q, nb = B, ng = G, op = "stats",
                       klo = as.integer(ceiling(k[1] / mult)), khi = as.integer(ceiling(k[2] / mult)))
  sdfac <- sqrt(mult * (B - 1) / (mult * B - 1))
  ctx_u <- ctx_text[rep_row]
  res <- dplyr::tibble(target = rep(glab, times = length(ctx_u)),
                       context = rep(ctx_u, each = G),
                       std.error = c(sweep(st$sd, 2, sdfac, "*")), value = c(st$mean),
                       lower.ci = c(st$lo), upper.ci = c(st$hi))
  res <- res[ctm_order(res$target, res$context), ]
  res %>%
    dplyr::group_by(target) %>%
    dplyr::slice_max(order_by = value, n = N) %>%
    dplyr::mutate(rank = 1:dplyr::n()) %>%
    dplyr::ungroup() %>%
    dplyr::select('target', 'context', 'rank', 'value', 'std.error', 'lower.ci', 'upper.ci')
}

# ------------------------------------------------------------------ get_nns_ratio
# bootstrap summaries for the union features only (the original filters to these afterwards)
ctm_boot_nns_ratio <- function(x_dem, numerator, denominator, union_nns, pre_trained, num_bootstraps, confidence_level) {
  groups <- x_dem@docvars$group
  lev <- ctm_group_levels(groups)
  if (is.null(lev) || length(lev) != 2) return(NULL)
  ord <- c(numerator, denominator)
  idx_list <- replicate(num_bootstraps, ctm_dem_sample_idx(x_dem, size = 1, replace = TRUE, by = groups), simplify = FALSE)
  Q <- ctm_group_means_batch(as.matrix(x_dem), idx_list, lapply(idx_list, function(i) groups[i]), ord)
  feats <- sort(union_nns, method = "radix")
  V <- pre_trained[feats, , drop = FALSE]
  k <- ctm_kidx(confidence_level, num_bootstraps)
  st <- ctm_sim_reduce(V, Q, nb = num_bootstraps, ng = 2L, ratio = TRUE, op = "stats", klo = k[1], khi = k[2])
  dplyr::tibble(feature = feats, value = st$mean[3, ], std.error = st$sd[3, ],
                lower.ci = st$lo[3, ], upper.ci = st$hi[3, ]) %>%
    dplyr::arrange(-value)
}

# list of permutation data frames (feature, value) restricted to the union features
ctm_perm_nns_ratio <- function(x, x_dem, groups, numerator, denominator, union_nns, pre_trained, num_permutations) {
  dids <- as.character(quanteda::docid(x))
  if (anyDuplicated(dids) || length(groups) != length(dids)) return(NULL)
  included <- match(rownames(x_dem), dids)
  if (anyNA(included)) return(NULL)
  ord <- c(numerator, denominator)
  labs <- replicate(num_permutations, sample(groups)[included], simplify = FALSE)
  if (any(vapply(labs, function(l) length(unique(l)) != 2, logical(1))))
    stop('nns_ratio can only be applied to a pair of embeddings i.e. nrow(x) must equal 2')
  idx <- rep(list(seq_len(nrow(x_dem))), num_permutations)
  Q <- ctm_group_means_batch(as.matrix(x_dem), idx, labs, ord)
  V <- pre_trained[union_nns, , drop = FALSE]
  S <- ctm_sim2(Q, V)
  lapply(seq_len(num_permutations), function(b)
    data.frame(feature = union_nns, value = S[2L * b - 1L, ] / S[2L * b, ]))
}

# ------------------------------------------------------------------ contrast_nns
ctm_contrast_boot <- function(E1, E2, V, num_bootstraps) {
  idx <- vector("list", num_bootstraps)
  for (b in seq_len(num_bootstraps)) {
    s1 <- sample(1:nrow(E1), replace = TRUE)
    s2 <- sample(1:nrow(E2), replace = TRUE)
    idx[[b]] <- c(s1, nrow(E1) + s2)
  }
  labs <- lapply(idx, function(i) ifelse(i <= nrow(E1), 1L, 2L))
  Q <- ctm_group_means_batch(rbind(E1, E2), idx, labs, 1:2)
  list(Q = Q)
}

ctm_contrast_perm <- function(E1, E2, num_permutations) {
  num_obs1 <- nrow(E1); num_obs <- num_obs1 + nrow(E2)
  idx <- vector("list", num_permutations); labs <- vector("list", num_permutations)
  for (b in seq_len(num_permutations)) {
    obs1_perm <- sample(1:num_obs, num_obs1)
    obs2_perm <- setdiff(1:num_obs, obs1_perm)
    shuffle <- sample(1:num_obs)
    idx[[b]] <- c(shuffle[obs1_perm], shuffle[obs2_perm])
    labs[[b]] <- rep(1:2, c(length(obs1_perm), length(obs2_perm)))
  }
  ctm_group_means_batch(rbind(E1, E2), idx, labs, 1:2)
}

# ------------------------------------------------------------------ bootstrap_nns
ctm_bootstrap_nns <- function(target_embeddings, V, num_bootstraps, confidence_level) {
  idx <- replicate(num_bootstraps, sample(1:nrow(target_embeddings), replace = TRUE), simplify = FALSE)
  Q <- ctm_group_means_batch(target_embeddings, idx, lapply(idx, function(i) rep(1L, length(i))), 1L)
  k <- ctm_kidx(confidence_level, num_bootstraps)
  ctm_sim_reduce(V, Q, nb = num_bootstraps, ng = 1L, op = "stats", klo = k[1], khi = k[2])
}
