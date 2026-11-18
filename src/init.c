#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern SEXP C_cluster_ols(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_jackknife(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_sim_reduce_cpu(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_metal_info(void);
extern SEXP C_metal_gemm(SEXP, SEXP, SEXP, SEXP);
extern SEXP C_metal_spmm(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_metal_sim_reduce(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"C_cluster_ols", (DL_FUNC) &C_cluster_ols, 6},
  {"C_jackknife", (DL_FUNC) &C_jackknife, 7},
  {"C_sim_reduce_cpu", (DL_FUNC) &C_sim_reduce_cpu, 10},
  {"C_metal_info", (DL_FUNC) &C_metal_info, 0},
  {"C_metal_gemm", (DL_FUNC) &C_metal_gemm, 4},
  {"C_metal_spmm", (DL_FUNC) &C_metal_spmm, 5},
  {"C_metal_sim_reduce", (DL_FUNC) &C_metal_sim_reduce, 9},
  {NULL, NULL, 0}
};

void R_init_FastconText(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
