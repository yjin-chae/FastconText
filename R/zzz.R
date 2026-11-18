.onAttach <- function(libname, pkgname) {
  pd <- utils::packageDescription(pkgname)
  mi <- tryCatch(metal_info(), error = function(e) list(available = FALSE, reason = conditionMessage(e)))
  gpu <- if (isTRUE(mi$available)) paste0("Metal GPU: ", mi$device) else paste0("Metal GPU not available (", mi$reason, "); using CPU engine")
  packageStartupMessage(pkgname, " v", pd$Version, " (fast implementation of conText 3.0.1). ", gpu, ".")
}

.onLoad <- function(libname, pkgname) {
  if (identical(Sys.getenv("_R_CHECK_LIMIT_CORES_"), "TRUE")) {
    Sys.setenv(
      "OMP_THREAD_LIMIT" = 2,
      "OMP_NUM_THREADS" = 1,
      "OPENBLAS_NUM_THREADS" = 1,
      "MKL_NUM_THREADS" = 1,
      "VECLIB_MAXIMUM_THREADS" = 1,
      "RCPP_PARALLEL_NUM_THREADS" = 1
    )
  }
}
