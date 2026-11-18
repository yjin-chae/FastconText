
# Largest absolute difference between FastconText and conText outputs, per task.
# Run from the benchmarks/ folder after bench_sample_data.R has written out/<config>/*.rds.
ws <- "."
num_diff <- function(a, b) {
  if (is.list(a) && !is.data.frame(a)) return(max(mapply(num_diff, a, b)))
  a <- as.data.frame(a); b <- as.data.frame(b)
  stopifnot(identical(dim(a), dim(b)))
  d <- 0
  for (j in seq_along(a)) {
    if (is.numeric(a[[j]])) { x <- abs(a[[j]] - b[[j]]); x <- x[!(is.na(a[[j]]) & is.na(b[[j]]))]
      if (anyNA(x)) return(Inf); d <- max(d, x, 0) }
    else if (!identical(as.character(a[[j]]), as.character(b[[j]]))) return(Inf)
  }
  d
}
tasks <- sub("\\.rds$", "", list.files(file.path(ws, "out/context"), "\\.rds$"))
res <- do.call(rbind, lapply(tasks, function(t) {
  o <- readRDS(file.path(ws, "out/context", paste0(t, ".rds")))
  data.frame(task = t,
    cpu_vs_conText = num_diff(o, readRDS(file.path(ws, "out/fast_cpu", paste0(t, ".rds")))),
    gpu_vs_conText = num_diff(o, readRDS(file.path(ws, "out/fast_metal", paste0(t, ".rds")))))
}))
pf <- file.path(ws, "out/context_parallel/reg_jackknife_permute.rds")
if (file.exists(pf)) res$par_vs_seq <- ifelse(res$task == "reg_jackknife_permute",
  num_diff(readRDS(file.path(ws, "out/context/reg_jackknife_permute.rds")), readRDS(pf)), NA)
print(res, digits = 3)
write.csv(res, file.path(ws, "agreement.csv"), row.names = FALSE)
