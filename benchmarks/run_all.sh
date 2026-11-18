#!/bin/sh
# Runs every configuration in sequence (about 10 minutes, almost all of it conText's one-core jackknife).
cd "$(dirname "$0")"
Rscript bench_sample_data.R FastconText metal out/fast_metal
Rscript bench_sample_data.R FastconText cpu   out/fast_cpu
Rscript bench_sample_data.R conText original  out/context
Rscript bench_sample_data.R conText parallel  out/context_parallel reg_jackknife_permute
Rscript compare_outputs.R
# Combine the per-configuration timings into timings.csv, then redraw the figure.
Rscript -e 'f <- file.path("out", c("context", "context_parallel", "fast_cpu", "fast_metal"), "timings.csv"); write.csv(do.call(rbind, lapply(f, read.csv)), "timings.csv", row.names = FALSE, quote = FALSE)'
python plot_speedup.py
