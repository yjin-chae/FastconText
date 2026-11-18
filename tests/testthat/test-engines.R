sim_data <- function(n = 200, D = 6, G = 30, seed = 1) {
  set.seed(seed)
  X <- data.frame(a = rbinom(n, 1, 0.5), b = rnorm(n))
  Y <- matrix(rnorm(n * D), n, D) + 0.4 * X$a
  ids <- factor(sample(seq_len(G), n, TRUE))
  list(X = X, Y = Y, ids = ids)
}

test_that("closed-form clustered OLS matches estimatr::lm_robust (stata CR1)", {
  skip_if_not_installed("estimatr")
  d <- sim_data()
  for (ids in list(d$ids, factor(seq_len(nrow(d$X))))) {
    w <- 1/as.vector(table(ids)[ids])
    fit <- estimatr::lm_robust(d$Y ~ ., data = d$X, clusters = ids, se_type = "stata", weights = w)
    out <- FastconText:::run_ols(Y = d$Y, X = d$X, ids = ids)
    expect_equal(unname(out$betas), unname(coef(fit)), tolerance = 1e-10)
    se2 <- sapply(seq_len(ncol(d$Y)), function(j) fit$std.error[, j]^2)
    expect_equal(unname(out$beta_error_null), unname(rowSums(se2[-1, , drop = FALSE])), tolerance = 1e-10)
  }
})

test_that("closed-form jackknife equals brute-force leave-one-cluster-out refits", {
  d <- sim_data(n = 150, G = 25)
  ids <- as.numeric(d$ids)
  fast <- FastconText:::ctm_jackknife_partials(d$X, d$Y, ids)
  slow <- t(sapply(seq_len(max(ids)), function(i) {
    keep <- ids != i
    FastconText:::run_ols(Y = d$Y[keep, ], X = d$X[keep, ], ids = factor(ids[keep]))$normed_betas_deflated
  }))
  expect_equal(as.vector(fast), as.vector(slow), tolerance = 1e-9)
})

test_that("CPU reduction engine matches plain R (mean, sd, order statistics, counts)", {
  set.seed(2)
  V <- matrix(rnorm(400 * 10), 400, 10); nb <- 50; ng <- 2
  Q <- matrix(rnorm(nb * ng * 10), nb * ng, 10)
  S <- text2vec::sim2(Q, V, method = "cosine", norm = "l2")
  st <- FastconText:::ctm_sim_reduce(V, Q, nb, ng, ratio = TRUE, op = "stats", klo = 3L, khi = 47L)
  g1 <- S[seq(1, nb * ng, 2), ]; g2 <- S[seq(2, nb * ng, 2), ]; rt <- g1 / g2
  expect_equal(st$mean[1, ], colMeans(g1), tolerance = 1e-12)
  expect_equal(st$sd[3, ], apply(rt, 2, sd), tolerance = 1e-12)
  expect_equal(st$lo[2, ], apply(g2, 2, function(x) sort(x)[3]), tolerance = 1e-12)
  expect_equal(st$hi[3, ], apply(rt, 2, function(x) sort(x)[47]), tolerance = 1e-12)
  thr <- rbind(colMeans(g1), colMeans(g2), abs(colMeans(rt) - 1))
  cnt <- FastconText:::ctm_sim_reduce(V, Q, nb, ng, ratio = TRUE, op = "count", thr = thr)
  expect_equal(cnt[3, ], colSums(abs(rt - 1) >= rep(thr[3, ], each = nb)))
})

test_that("Metal engine agrees with the CPU engine", {
  skip_if_not(metal_available(), "no Metal device visible")
  set.seed(3)
  A <- matrix(rnorm(300 * 70), 300); B <- matrix(rnorm(70 * 45), 70)
  expect_equal(.Call(FastconText:::C_metal_gemm, A, B, FALSE, FALSE), A %*% B, tolerance = 1e-5)
  expect_equal(.Call(FastconText:::C_metal_gemm, t(A), t(B), TRUE, TRUE), A %*% B, tolerance = 1e-5)
  V <- matrix(rnorm(3000 * 50), 3000); Q <- matrix(rnorm(200 * 50), 200)
  old <- options(FastconText.backend = "metal"); on.exit(options(old))
  gpu <- FastconText:::ctm_sim_reduce(V, Q, 100, 2, ratio = TRUE, op = "stats", klo = 5L, khi = 95L)
  options(FastconText.backend = "cpu")
  cpu <- FastconText:::ctm_sim_reduce(V, Q, 100, 2, ratio = TRUE, op = "stats", klo = 5L, khi = 95L)
  expect_equal(gpu$mean[1:2, ], cpu$mean[1:2, ], tolerance = 1e-5)
  expect_equal(gpu$lo[1:2, ], cpu$lo[1:2, ], tolerance = 1e-5)
  d <- sim_data(n = 300, G = 40)
  options(FastconText.backend = "auto", FastconText.jackknife = "metal")
  jg <- FastconText:::ctm_jackknife_partials(d$X, d$Y, as.numeric(d$ids))
  options(FastconText.jackknife = "cpu")
  jc <- FastconText:::ctm_jackknife_partials(d$X, d$Y, as.numeric(d$ids))
  expect_equal(attr(jg, "backend"), "metal")
  expect_equal(as.vector(jg), as.vector(jc), tolerance = 1e-9)
})

test_that("rank-deficient leave-out fits: default matches the original code; 'na' blanks the collinear set", {
  skip_if_not_installed("estimatr")
  set.seed(4)
  n <- 30
  X <- data.frame(x1 = rnorm(n), lev_b = 0, lev_c = 0)
  lev <- c("a", sample(c("b", "c", "d"), n - 1, TRUE))      # reference level "a" has one context
  X$lev_b <- as.numeric(lev == "b"); X$lev_c <- as.numeric(lev == "c"); X$lev_d <- as.numeric(lev == "d")
  Y <- matrix(rnorm(n * 5), n, 5) + 0.5 * X$lev_b
  ids <- factor(seq_len(n))
  theta <- FastconText:::run_ols(Y = Y, X = X, ids = ids)$normed_betas_deflated
  orig <- suppressWarnings(FastconText:::run_jackknife_orig(theta, X, Y, ids, 0.95))
  expect_warning(fast <- FastconText:::run_jackknife(theta, X, Y, ids, 0.95), "rank-deficient")
  expect_equal(as.matrix(fast), as.matrix(orig), tolerance = 1e-9)
  old <- options(FastconText.jackknife_rank = "na"); on.exit(options(old))
  na <- suppressWarnings(FastconText:::run_jackknife(theta, X, Y, ids, 0.95))
  expect_true(all(is.na(na[c("lev_b", "lev_c", "lev_d"), "std.error"])))
  expect_false(is.na(na["x1", "std.error"]))
})
