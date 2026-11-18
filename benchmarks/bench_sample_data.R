# Benchmark of FastconText against conText 3.0.1 on the example data shipped with conText
# (cr_sample_corpus, cr_glove_subset, cr_transform). Tasks follow conText's help-page examples.
# Usage: Rscript bench_sample_data.R <pkg> <engine> <outdir> [task1,task2,...]
#   pkg = conText | FastconText
#   engine = original (conText default, one core) | parallel (conText, parallel = TRUE with doParallel)
#            | cpu | metal (FastconText engines)
# Each task is run once as a warm-up and then 3 times; the median is reported
# (the one-core conText jackknife task is run once, without warm-up).
args <- commandArgs(TRUE); pkg <- args[1]; engine <- args[2]; out <- args[3]
only <- if (length(args) >= 4) strsplit(args[4], ",")[[1]] else NULL
suppressPackageStartupMessages({ library(quanteda); library(pkg, character.only = TRUE) })
if (pkg == "FastconText") options(FastconText.backend = engine,
                                   FastconText.jackknife = if (engine == "metal") "metal" else "cpu")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
info <- if (pkg == "FastconText") metal_info() else list(available = NA, device = NA)
cat(pkg, engine, "GPU:", info$available, info$device, "\n")
PAR <- FALSE
if (pkg == "conText" && engine == "parallel") {   # conText's own parallel jackknife (foreach backend)
  ncore <- parallel::detectCores(logical = FALSE)
  doParallel::registerDoParallel(cores = ncore); PAR <- TRUE
  cat("doParallel workers:", foreach::getDoParWorkers(), "\n")
}
if (pkg == "FastconText") cat("FastconText threads:", FastconText:::ctm_threads(), "\n")

toks <- tokens(cr_sample_corpus)
ctx  <- tokens_context(x = toks, pattern = "immigration", window = 6L, verbose = FALSE)
set.seed(2022L); s50  <- tokens_sample(ctx, size = 50,  by = docvars(ctx, "party"))
set.seed(2022L); s100 <- tokens_sample(ctx, size = 100, by = docvars(ctx, "party"))
ctx_k <- tokens_context(x = toks, pattern = "immigration", window = 6L, rm_keyword = FALSE, verbose = FALSE)
set.seed(2022L); s100k <- tokens_sample(ctx_k, size = 100, by = docvars(ctx_k, "party"))
feats50 <- featnames(dfm(s50))
gc_ctx <- get_context(x = cr_sample_corpus, target = "immigration", window = 6, valuetype = "fixed",
                      case_insensitive = TRUE, hard_cut = FALSE, verbose = FALSE)
lv <- get_local_vocab(gc_ctx$context, pre_trained = cr_glove_subset)
G <- cr_glove_subset; Tm <- cr_transform

tasks <- list(
  reg_permute = function() { set.seed(2021L); m <- conText(immigration ~ party + gender, data = toks, pre_trained = G,
      transform = TRUE, transform_matrix = Tm, jackknife = FALSE, permute = TRUE, num_permutations = 100,
      window = 6, case_insensitive = TRUE, verbose = FALSE); m@normed_coefficients },
  reg_jackknife_permute = function() { set.seed(2021L); m <- conText(immigration ~ party + gender, data = toks, pre_trained = G,
      transform = TRUE, transform_matrix = Tm, jackknife = TRUE, parallel = PAR, permute = TRUE, num_permutations = 100,
      window = 6, case_insensitive = TRUE, verbose = FALSE); m@normed_coefficients },
  get_nns = function() { set.seed(2021L); get_nns(x = s50, N = 10, groups = docvars(s50, "party"), candidates = feats50,
      pre_trained = G, transform = TRUE, transform_matrix = Tm, bootstrap = TRUE, num_bootstraps = 100,
      stem = FALSE, as_list = FALSE) },
  get_cos_sim = function() { set.seed(2021L); get_cos_sim(x = s100, groups = docvars(s100, "party"),
      features = c("reform", "enforce"), pre_trained = G, transform = TRUE, transform_matrix = Tm,
      bootstrap = TRUE, num_bootstraps = 100, stem = FALSE, as_list = FALSE) },
  get_ncs = function() { set.seed(2021L); get_ncs(x = s100k, N = 10, groups = docvars(s100k, "party"), pre_trained = G,
      transform = TRUE, transform_matrix = Tm, bootstrap = TRUE, num_bootstraps = 100, as_list = FALSE) },
  get_nns_ratio = function() { set.seed(2021L); get_nns_ratio(x = s50, N = 10, groups = docvars(s50, "party"),
      numerator = "R", candidates = feats50, pre_trained = G, transform = TRUE, transform_matrix = Tm,
      bootstrap = TRUE, num_bootstraps = 100, permute = TRUE, num_permutations = 100, verbose = FALSE) },
  contrast_nns = function() { set.seed(42L); contrast_nns(x = s100, groups = docvars(s100, "party"), pre_trained = G,
      transform = TRUE, transform_matrix = Tm, bootstrap = TRUE, num_bootstraps = 100, permute = TRUE,
      num_permutations = 100, candidates = NULL, N = 20, verbose = FALSE) },
  bootstrap_nns = function() { set.seed(42L); bootstrap_nns(context = gc_ctx$context, pre_trained = G,
      transform_matrix = Tm, transform = TRUE, candidates = lv, bootstrap = TRUE, num_bootstraps = 100, N = 50, norm = "l2") },
  prototypical_context = function() prototypical_context(context = gc_ctx$context, pre_trained = G,
      transform = TRUE, transform_matrix = Tm, N = 3, norm = "l2")
)
tim <- data.frame()
for (nm in if (is.null(only)) names(tasks) else only) {
  reps <- if (pkg == "conText" && nm == "reg_jackknife_permute") 1 else 3
  if (reps > 1) invisible(tasks[[nm]]())   # warm-up (GPU shader compile, caches) - excluded from timing
  el <- numeric(reps)
  for (r in seq_len(reps)) { t0 <- proc.time()[["elapsed"]]; res <- tasks[[nm]](); el[r] <- proc.time()[["elapsed"]] - t0 }
  saveRDS(res, file.path(out, paste0(nm, ".rds")))
  tim <- rbind(tim, data.frame(pkg = pkg, engine = engine, task = nm, reps = reps, median_s = median(el), min_s = min(el)))
  cat(sprintf("%-24s %10.3f s\n", nm, median(el)))
}
write.csv(tim, file.path(out, "timings.csv"), row.names = FALSE)
cat("DONE\n")
