# Extended formulas in conText() (new in FastconText).
#
# conText 3.0.1 accepts only `+` between plain variable names. FastconText also accepts anything
# lm() accepts on the right-hand side: interactions (`*`, `:`, `^`, `/`, `%in%`), term removal (`-`),
# and function calls (`log(x)`, `I(x^2)`, `factor(year)`, `poly(x, 2)`, `splines::ns(x, 3)`, ...).
# The design is built by stats::model.frame() and stats::model.matrix(), so terms are expanded and
# coded exactly as in lm(), with two conText conventions kept:
#   * categorical variables (character, factor, ordered factor, logical) use treatment coding, with
#     columns named `variable_level` (e.g. `party_R`), as fastDummies names them in conText;
#   * rows with missing covariates are kept (NA), as in conText, unless
#     options(FastconText.na_action = "drop") (then they are dropped listwise, as lm() does).
# Columns that are linear combinations of earlier ones (e.g. the four cells of `party:gender`
# next to the intercept, or an empty cell) are dropped with a message; within a term the first
# column (the reference cell) is the one dropped. Formulas made only of `+` and plain names, and
# `~ .`, never reach this code.

ctm_formula_extended <- function(rhs) {
  if (identical(rhs, quote(.))) return(FALSE)
  length(setdiff(all.names(rhs), c("+", all.vars(rhs)))) > 0
}

ctm_formula_design <- function(rhs, docvars, env = parent.frame(), cluster_variable = NULL, na_drop = FALSE) {
  dv <- as.data.frame(docvars)
  cl_ok <- if (is.null(cluster_variable)) rep(TRUE, nrow(dv)) else !is.na(dv[[cluster_variable]])
  dv <- dv[, setdiff(names(dv), cluster_variable), drop = FALSE]
  tt <- stats::terms(stats::as.formula(call("~", rhs), env = env), data = dv)
  if (attr(tt, "intercept") == 0) stop("conText always includes an intercept; remove `- 1` or `+ 0` from \"formula\".", call. = FALSE)
  if (!is.null(attr(tt, "offset"))) stop("offset() is not supported in \"formula\".", call. = FALSE)
  if (length(attr(tt, "term.labels")) == 0) stop("\"formula\" has no covariates.", call. = FALSE)
  mf <- stats::model.frame(tt, data = dv, na.action = stats::na.pass)
  # options(FastconText.na_action = "drop"): listwise deletion after the variables are evaluated, as
  # lm()'s na.omit does; factor() below then drops levels left without rows
  ok <- NULL
  if (na_drop) {
    ok <- stats::complete.cases(mf) & cl_ok
    if (all(ok)) ok <- NULL else { mf <- mf[ok, , drop = FALSE]; dv <- dv[ok, , drop = FALSE] }
  }
  # conText naming and coding for categorical variables: treatment contrasts, `variable_level`
  for (v in names(mf)) {
    x <- mf[[v]]
    if (is.character(x) || is.factor(x) || is.logical(x)) {
      f <- factor(x, ordered = FALSE)
      levels(f) <- paste0("_", levels(f))
      mf[[v]] <- f
    }
  }
  ca <- lapply(Filter(is.factor, mf), function(z) "contr.treatment")
  mm <- stats::model.matrix(tt, mf, contrasts.arg = if (length(ca)) ca else NULL)
  if (nrow(mm) != nrow(dv)) stop("could not build the covariates from \"formula\" (rows were lost).", call. = FALSE)
  assign <- attr(mm, "assign")
  # drop aliased columns (structural, e.g. `party:gender` alone, or empty cells)
  cc <- stats::complete.cases(mm)
  keep <- seq_len(ncol(mm))
  if (any(cc) && qr(mm[cc, , drop = FALSE])$rank < ncol(mm)) {
    keep <- which(assign == 0)
    for (j in setdiff(unique(assign), 0)) {
      cols <- which(assign == j)
      for (k in c(cols[-1], cols[1])) if (qr(mm[cc, c(keep, k), drop = FALSE])$rank > length(keep)) keep <- c(keep, k)
    }
    keep <- sort(keep)
    dropped <- setdiff(colnames(mm), colnames(mm)[keep])
    message("conText: dropped ", length(dropped), " covariate column(s) that are linear combinations of the others: ",
            paste0(dropped, collapse = ", "))
  }
  keep <- setdiff(keep, which(assign == 0))
  out <- as.data.frame(mm[, keep, drop = FALSE], check.names = FALSE)
  rownames(out) <- NULL
  attr(out, "design_names") <- names(out)
  attr(out, "keep_rows") <- ok
  out
}

# model.matrix()/lm_robust backtick non-syntactic names such as `party_R:gender_M` or `log(x)`; drop them
ctm_clean_terms <- function(fit, design_names) {
  if (length(design_names) == 0) return(fit)
  fix <- function(nm) { hit <- nm %in% paste0("`", design_names, "`"); nm[hit] <- substr(nm[hit], 2, nchar(nm[hit]) - 1); nm }
  rownames(fit$betas) <- fix(rownames(fit$betas))
  for (k in c("normed_betas_orig", "normed_betas_deflated", "beta_error_null")) names(fit[[k]]) <- fix(names(fit[[k]]))
  fit
}
