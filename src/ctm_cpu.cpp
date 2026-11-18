// FastconText CPU engine (double precision, multi-threaded).
//  * C_cluster_ols      : weighted OLS + Stata CR1 cluster-robust SEs for D outcomes at once
//                         (identical to estimatr::lm_robust(se_type = "stata") used by conText).
//  * C_jackknife        : closed-form leave-one-cluster-out jackknife of the debiased squared norms.
//  * C_sim_reduce_cpu   : batched cosine similarities + per-column bootstrap/permutation reductions.
#include "ctm.h"
#include <thread>
#include <algorithm>
#include <cmath>
#include <limits>

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#ifndef FCONE
#define FCONE
#endif

// ---------------------------------------------------------------- helpers
template <class F>
static void parallel_for(long n, int nthreads, F fn) {
  if (nthreads < 1) nthreads = 1;
  if (n < 2 * nthreads || nthreads == 1) { fn(0L, n); return; }
  std::vector<std::thread> th;
  long chunk = (n + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    long b = t * chunk, e = std::min(n, b + chunk);
    if (b >= e) break;
    th.emplace_back([=]() { fn(b, e); });
  }
  for (auto &x : th) x.join();
}

// Gauss-Jordan inverse with partial pivoting (row-major p x p). Returns false if singular.
static bool invert(const double *H, double *A, int p) {
  std::vector<double> M(H, H + p * p);
  for (int i = 0; i < p * p; ++i) A[i] = 0.0;
  for (int i = 0; i < p; ++i) A[i * p + i] = 1.0;
  double scale = 0.0;
  for (int i = 0; i < p; ++i) scale = std::max(scale, std::fabs(M[i * p + i]));
  if (scale == 0.0) return false;
  for (int c = 0; c < p; ++c) {
    int piv = c; double best = std::fabs(M[c * p + c]);
    for (int r = c + 1; r < p; ++r) if (std::fabs(M[r * p + c]) > best) { best = std::fabs(M[r * p + c]); piv = r; }
    if (best <= 1e-11 * scale) return false;
    if (piv != c) for (int k = 0; k < p; ++k) { std::swap(M[c * p + k], M[piv * p + k]); std::swap(A[c * p + k], A[piv * p + k]); }
    double inv = 1.0 / M[c * p + c];
    for (int k = 0; k < p; ++k) { M[c * p + k] *= inv; A[c * p + k] *= inv; }
    for (int r = 0; r < p; ++r) if (r != c) {
      double f = M[r * p + c]; if (f == 0.0) continue;
      for (int k = 0; k < p; ++k) { M[r * p + k] -= f * M[c * p + k]; A[r * p + k] -= f * A[c * p + k]; }
    }
  }
  return true;
}

// Order-preserving rank detection (as lm() / LINPACK dqrdc2): walk columns left to right and
// drop a column whose residual norm, after projecting out the kept columns, is below tol on the
// unit-normalised scale. Done on the Gram matrix via an incremental Cholesky factor.
int ctm_rank_keep(const double *H, int p, int *keep) {
  std::vector<double> sc(p), L((size_t)p * p, 0.0);
  std::vector<int> K;
  for (int c = 0; c < p; ++c) sc[c] = H[c * p + c] > 0 ? 1.0 / std::sqrt(H[c * p + c]) : 0.0;
  for (int c = 0; c < p; ++c) {
    keep[c] = 0;
    if (sc[c] == 0.0) continue;
    int r = (int)K.size();
    std::vector<double> l(r);
    for (int i = 0; i < r; ++i) {
      double v = H[K[i] * p + c] * sc[K[i]] * sc[c];
      for (int m = 0; m < i; ++m) v -= L[(size_t)i * p + m] * l[m];
      l[i] = v / L[(size_t)i * p + i];
    }
    double res = 1.0; for (int i = 0; i < r; ++i) res -= l[i] * l[i];
    if (!(res > CTM_RANK_TOL * CTM_RANK_TOL)) continue;
    for (int i = 0; i < r; ++i) L[(size_t)r * p + i] = l[i];
    L[(size_t)r * p + r] = std::sqrt(res);
    K.push_back(c); keep[c] = 1;
  }
  return (int)K.size();
}

static int as_threads(SEXP x) { int t = Rf_asInteger(x); return (t == NA_INTEGER || t < 1) ? 1 : t; }

// Full-sample weighted fit + per-cluster sufficient statistics (shared by OLS and jackknife).
// Xd: n x p (col-major, intercept included), Y: n x D, cl: 0-based cluster index, w: weights.
static bool fit_stats(const double *X, const double *Y, const int *cl, const double *w,
                      int n, int p, int D, int G, int nthreads, JKStats &S, bool need_moments) {
  S.n = n; S.p = p; S.D = D; S.G = G;
  S.H.assign(p * p, 0.0);
  std::vector<double> XtWY(p * D, 0.0);   // [d*p + a]
  std::vector<double> WX((size_t)n * p);  // col-major n x p
  for (int a = 0; a < p; ++a) for (int i = 0; i < n; ++i) WX[(size_t)a * n + i] = w[i] * X[(size_t)a * n + i];
  for (int a = 0; a < p; ++a) for (int b = 0; b < p; ++b) {
    double s = 0; for (int i = 0; i < n; ++i) s += WX[(size_t)a * n + i] * X[(size_t)b * n + i];
    S.H[a * p + b] = s;
  }
  parallel_for(D, nthreads, [&](long d0, long d1) {
    for (long d = d0; d < d1; ++d) for (int a = 0; a < p; ++a) {
      double s = 0; const double *y = Y + (size_t)d * n, *wx = WX.data() + (size_t)a * n;
      for (int i = 0; i < n; ++i) s += wx[i] * y[i];
      XtWY[d * p + a] = s;
    }
  });
  std::vector<double> A(p * p);
  if (!invert(S.H.data(), A.data(), p)) return false;
  S.B.assign((size_t)D * p, 0.0);
  for (int d = 0; d < D; ++d) for (int a = 0; a < p; ++a) {
    double s = 0; for (int b = 0; b < p; ++b) s += A[a * p + b] * XtWY[d * p + b];
    S.B[(size_t)d * p + a] = s;
  }
  // per-cluster X'WX and scores of full-sample residuals
  S.ng.assign(G, 0);
  S.Hg.assign((size_t)G * p * p, 0.0);
  S.Zg.assign((size_t)G * D * p, 0.0);
  for (int i = 0; i < n; ++i) {
    int g = cl[i]; S.ng[g]++;
    for (int a = 0; a < p; ++a) for (int b = 0; b < p; ++b)
      S.Hg[((size_t)g * p + a) * p + b] += WX[(size_t)a * n + i] * X[(size_t)b * n + i];
  }
  parallel_for(D, nthreads, [&](long d0, long d1) {
    std::vector<double> xi(p), wxi(p);
    for (int i = 0; i < n; ++i) {
      for (int a = 0; a < p; ++a) { xi[a] = X[(size_t)a * n + i]; wxi[a] = WX[(size_t)a * n + i]; }
      int g = cl[i];
      for (long d = d0; d < d1; ++d) {
        double e = Y[(size_t)d * n + i];
        const double *bd = S.B.data() + (size_t)d * p;
        for (int a = 0; a < p; ++a) e -= xi[a] * bd[a];
        double *z = S.Zg.data() + ((size_t)g * D + d) * p;
        for (int a = 0; a < p; ++a) z[a] += wxi[a] * e;
      }
    }
  });
  S.tss.assign(D, 0.0);
  {
    double sw = 0; for (int i = 0; i < n; ++i) sw += w[i];
    for (int d = 0; d < D; ++d) {
      const double *y = Y + (size_t)d * n; double m = 0;
      for (int i = 0; i < n; ++i) m += w[i] * y[i];
      m /= sw; double t = 0;
      for (int i = 0; i < n; ++i) t += w[i] * (y[i] - m) * (y[i] - m);
      S.tss[d] = t;
    }
  }
  if (!need_moments) return true;
  S.M1.assign((size_t)D * p * p, 0.0);
  S.M2.assign((size_t)D * p * p * p, 0.0);
  S.M4.assign((size_t)p * p * p * p, 0.0);
  parallel_for(D, nthreads, [&](long d0, long d1) {
    for (long d = d0; d < d1; ++d) {
      double *m1 = S.M1.data() + (size_t)d * p * p, *m2 = S.M2.data() + (size_t)d * p * p * p;
      for (int g = 0; g < G; ++g) {
        const double *z = S.Zg.data() + ((size_t)g * D + d) * p, *h = S.Hg.data() + (size_t)g * p * p;
        for (int a = 0; a < p; ++a) {
          for (int b = 0; b < p; ++b) m1[a * p + b] += z[a] * z[b];
          for (int bc = 0; bc < p * p; ++bc) m2[a * p * p + bc] += z[a] * h[bc];
        }
      }
    }
  });
  int p2 = p * p;
  for (int g = 0; g < G; ++g) {
    const double *h = S.Hg.data() + (size_t)g * p2;
    for (int u = 0; u < p2; ++u) for (int v = 0; v < p2; ++v) S.M4[u * p2 + v] += h[u] * h[v];
  }
  return true;
}

// ---------------------------------------------------------------- C_cluster_ols
// Returns list(B = p x D, se2 = p x D) with Stata CR1 small-sample factor G/(G-1)*(n-1)/(n-p).
extern "C" SEXP C_cluster_ols(SEXP Xd, SEXP Y, SEXP cl, SEXP G_, SEXP w, SEXP nthreads_) {
  int n = Rf_nrows(Xd), p = Rf_ncols(Xd), D = Rf_ncols(Y), G = Rf_asInteger(G_);
  int nt = as_threads(nthreads_);
  JKStats S;
  if (!fit_stats(REAL(Xd), REAL(Y), INTEGER(cl), REAL(w), n, p, D, G, nt, S, false))
    Rf_error("design matrix is singular");
  std::vector<double> A(p * p);
  invert(S.H.data(), A.data(), p);
  double c = (double)G / (G - 1.0) * (n - 1.0) / (double)(n - p);
  SEXP Bm = PROTECT(Rf_allocMatrix(REALSXP, p, D));
  SEXP Sm = PROTECT(Rf_allocMatrix(REALSXP, p, D));
  double *bo = REAL(Bm), *so = REAL(Sm);
  for (int d = 0; d < D; ++d) for (int a = 0; a < p; ++a) bo[a + (size_t)p * d] = S.B[(size_t)d * p + a];
  parallel_for(D, nt, [&](long d0, long d1) {
    std::vector<double> acc(p);
    for (long d = d0; d < d1; ++d) {
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int g = 0; g < G; ++g) {
        const double *z = S.Zg.data() + ((size_t)g * D + d) * p;
        for (int k = 0; k < p; ++k) { double t = 0; for (int a = 0; a < p; ++a) t += A[k * p + a] * z[a]; acc[k] += t * t; }
      }
      for (int k = 0; k < p; ++k) {
        double v = c * acc[k], ref = S.tss[d] / (double)(n - p) * A[k * p + k];
        so[k + (size_t)p * d] = (std::fabs(v) <= CTM_ZERO_TOL * ref) ? NA_REAL : (v < 0 ? R_NaN : v);
      }
    }
  });
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, Bm); SET_VECTOR_ELT(out, 1, Sm);
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(nm, 0, Rf_mkChar("B")); SET_STRING_ELT(nm, 1, Rf_mkChar("se2"));
  Rf_setAttrib(out, R_NamesSymbol, nm);
  UNPROTECT(4);
  return out;
}

// ---------------------------------------------------------------- jackknife (CPU per-cluster loop)
// out: G x (p-1), column-major; entry (j, k-1) = sum_d [ beta_{-j,kd}^2 - se^2_{-j,kd} ] (NA/NaN as estimatr).
// Leave-one-out fits that lose rank drop the later collinear column(s), as lm() does. estimatr picks the
// dropped column by pivoted QR instead, so run_jackknife() refits those leave-outs with the original code.
static void jk_one(const JKStats &S, long j, double *out) {
  const int p = S.p, D = S.D, G = S.G, n = S.n, p2 = p * p;
  std::vector<double> Hj(p2), Aj(p2, 0.0), aa(p2), K(p2), u(p), delta(p), P2(p), HB(p);
  std::vector<int> keep(p);
  const double *hj = S.Hg.data() + (size_t)j * p2;
  for (int i = 0; i < p2; ++i) Hj[i] = S.H[i] - hj[i];
  int r = ctm_rank_keep(Hj.data(), p, keep.data());
  std::vector<int> Kx; for (int c = 0; c < p; ++c) if (keep[c]) Kx.push_back(c);
  {
    std::vector<double> Hk((size_t)r * r), Ak((size_t)r * r);
    for (int x = 0; x < r; ++x) for (int y = 0; y < r; ++y) Hk[x * r + y] = Hj[Kx[x] * p + Kx[y]];
    if (r == 0 || !invert(Hk.data(), Ak.data(), r)) { for (int k = 1; k < p; ++k) out[j + (size_t)G * (k - 1)] = NA_REAL; return; }
    for (int x = 0; x < r; ++x) for (int y = 0; y < r; ++y) Aj[Kx[x] * p + Kx[y]] = Ak[x * r + y];
  }
  bool full = (r == p);
  double Gj = G - 1.0, Nj = (double)n - S.ng[j];
  double cj = Gj / (Gj - 1.0) * (Nj - 1.0) / (Nj - r);
  for (int k = 1; k < p; ++k) {
    if (!keep[k]) { out[j + (size_t)G * (k - 1)] = NA_REAL; continue; }
    const double *a = Aj.data() + k * p;
    for (int x = 0; x < p; ++x) for (int y = 0; y < p; ++y) aa[x * p + y] = a[x] * a[y];
    for (int c = 0; c < p; ++c) for (int c2 = 0; c2 < p; ++c2) {
      double s = 0;
      for (int b = 0; b < p; ++b) { if (a[b] == 0.0) continue; for (int b2 = 0; b2 < p; ++b2) s += aa[b * p + b2] * S.M4[(size_t)(b * p + c) * p2 + b2 * p + c2]; }
      K[c * p + c2] = s;
    }
    for (int c = 0; c < p; ++c) { double s = 0; for (int b = 0; b < p; ++b) s += a[b] * hj[b * p + c]; u[c] = s; }
    double acc = 0.0; bool missing = false, nan = false;
    for (int d = 0; d < D; ++d) {
      const double *zj = S.Zg.data() + ((size_t)j * D + d) * p, *Bd = S.B.data() + (size_t)d * p;
      // leave-out shift: Delta_K = A_K (Hj[K,drop] B_drop - z_j[K]),  Delta_drop = -B_drop
      if (full) {
        for (int x = 0; x < p; ++x) { double s = 0; for (int y = 0; y < p; ++y) s -= Aj[x * p + y] * zj[y]; delta[x] = s; }
      } else {
        for (int x = 0; x < p; ++x) { double s = -zj[x]; for (int y = 0; y < p; ++y) if (!keep[y]) s += Hj[x * p + y] * Bd[y]; HB[x] = s; }
        for (int x = 0; x < p; ++x) {
          if (keep[x]) { double s = 0; for (int y = 0; y < p; ++y) s += Aj[x * p + y] * HB[y]; delta[x] = s; }
          else delta[x] = -Bd[x];
        }
      }
      const double *m1 = S.M1.data() + (size_t)d * p2, *m2 = S.M2.data() + (size_t)d * p2 * p;
      double t1 = 0; for (int v = 0; v < p2; ++v) t1 += aa[v] * m1[v];
      for (int c = 0; c < p; ++c) P2[c] = 0.0;
      for (int ab = 0; ab < p2; ++ab) { double f = aa[ab]; if (f == 0.0) continue; const double *row = m2 + ab * p; for (int c = 0; c < p; ++c) P2[c] += f * row[c]; }
      double t2 = 0; for (int c = 0; c < p; ++c) t2 += P2[c] * delta[c];
      double t3 = 0; for (int x = 0; x < p; ++x) { double s = 0; for (int y = 0; y < p; ++y) s += K[x * p + y] * delta[y]; t3 += delta[x] * s; }
      double az = 0, ud = 0; for (int x = 0; x < p; ++x) { az += a[x] * zj[x]; ud += u[x] * delta[x]; }
      double own = (az - ud) * (az - ud);
      double inner = t1 - 2.0 * t2 + t3 - own;
      double mag = std::fabs(t1) + 2.0 * std::fabs(t2) + std::fabs(t3) + own;
      if (std::fabs(inner) < CTM_CANCEL_TOL * mag) {
        // influential leave-out: the expanded quadratic form cancels; sum the leave-out scores directly
        inner = 0.0;
        for (int g = 0; g < G; ++g) {
          if (g == j) continue;
          const double *zg = S.Zg.data() + ((size_t)g * D + d) * p, *hg = S.Hg.data() + (size_t)g * p2;
          double s = 0;
          for (int b = 0; b < p; ++b) {
            if (a[b] == 0.0) continue;
            double hd = 0; for (int c2 = 0; c2 < p; ++c2) hd += hg[b * p + c2] * delta[c2];
            s += a[b] * (zg[b] - hd);
          }
          inner += s * s;
        }
      }
      double se2 = cj * inner;
      double ref = S.tss[d] / (Nj - r) * a[k];
      if (std::fabs(se2) <= CTM_ZERO_TOL * ref) missing = true;
      else if (se2 < 0) nan = true;
      double bn = Bd[k] + delta[k];
      acc += bn * bn - se2;
    }
    out[j + (size_t)G * (k - 1)] = missing ? NA_REAL : (nan ? R_NaN : acc);
  }
}

static void jackknife_cpu(const JKStats &S, const std::vector<long> &js, double *out, int nthreads) {
  parallel_for((long)js.size(), nthreads, [&](long i0, long i1) { for (long i = i0; i < i1; ++i) jk_one(S, js[i], out); });
}

// backend: 0 = CPU double, 1 = Metal double-float. Returns G x (p-1) matrix (attr "backend").
extern "C" SEXP C_jackknife(SEXP Xd, SEXP Y, SEXP cl, SEXP G_, SEXP w, SEXP nthreads_, SEXP backend_) {
  int n = Rf_nrows(Xd), p = Rf_ncols(Xd), D = Rf_ncols(Y), G = Rf_asInteger(G_);
  int nt = as_threads(nthreads_), backend = Rf_asInteger(backend_);
  if (p < 2) Rf_error("jackknife needs at least one covariate");
  JKStats S;
  if (!fit_stats(REAL(Xd), REAL(Y), INTEGER(cl), REAL(w), n, p, D, G, nt, S, true))
    Rf_error("design matrix is singular");
  SEXP out = PROTECT(Rf_allocMatrix(REALSXP, G, p - 1));
  const char *used = "cpu";
  std::string err;
  std::vector<long> all(G); for (int j = 0; j < G; ++j) all[j] = j;
  if (backend == 1) {
    // rank-deficient leave-out fits are routed to the CPU; the GPU flags degenerate variances for redo
    std::vector<unsigned char> skip(G, 0), redo(G, 0);
    std::vector<int> keep(p); std::vector<double> Hj(p * p);
    for (int j = 0; j < G; ++j) {
      for (int i = 0; i < p * p; ++i) Hj[i] = S.H[i] - S.Hg[(size_t)j * p * p + i];
      if (ctm_rank_keep(Hj.data(), p, keep.data()) < p) skip[j] = 1;
    }
    if (ctm_metal_jackknife(S, skip, REAL(out), redo, err)) {
      used = "metal";
      std::vector<long> js; for (int j = 0; j < G; ++j) if (skip[j] || redo[j]) js.push_back(j);
      jackknife_cpu(S, js, REAL(out), nt);
    } else {
      Rf_warning("Metal jackknife unavailable (%s); using CPU", err.c_str());
      jackknife_cpu(S, all, REAL(out), nt);
    }
  } else jackknife_cpu(S, all, REAL(out), nt);
  Rf_setAttrib(out, Rf_install("backend"), Rf_mkString(used));
  UNPROTECT(1);
  return out;
}

// ---------------------------------------------------------------- batched similarity reductions (CPU)
// V: Vtot x D vocabulary, Q: R x D queries with R = nb*ng (row r = b*ng + g).
// Rows are L2-normalised internally (text2vec::sim2(norm = "l2") semantics).
// op 0 = stats  -> list(mean, sd, lo, hi), each nq x Vtot (nq = ng + ratio)
// op 1 = count  -> nq x Vtot counts of (S >= thr) for groups and (|S1/S2 - 1| >= thr) for ratio
// op 2 = full   -> R x Vtot similarity matrix
static void l2rows(const double *X, int nr, int nc, std::vector<double> &out) {
  out.assign((size_t)nr * nc, 0.0);
  std::vector<double> ss(nr, 0.0);
  for (int c = 0; c < nc; ++c) for (int r = 0; r < nr; ++r) { double v = X[(size_t)c * nr + r]; ss[r] += v * v; }
  for (int r = 0; r < nr; ++r) ss[r] = 1.0 / std::sqrt(ss[r]);
  for (int c = 0; c < nc; ++c) for (int r = 0; r < nr; ++r) out[(size_t)c * nr + r] = X[(size_t)c * nr + r] * ss[r];
}

extern "C" SEXP C_sim_reduce_cpu(SEXP V_, SEXP Q_, SEXP nb_, SEXP ng_, SEXP ratio_, SEXP op_,
                                 SEXP klo_, SEXP khi_, SEXP thr_, SEXP nthreads_) {
  int Vtot = Rf_nrows(V_), D = Rf_ncols(V_), R = Rf_nrows(Q_);
  int nb = Rf_asInteger(nb_), ng = Rf_asInteger(ng_), ratio = Rf_asLogical(ratio_), op = Rf_asInteger(op_);
  int nt = as_threads(nthreads_);
  if (Rf_ncols(Q_) != D || R != nb * ng) Rf_error("dimension mismatch in sim_reduce");
  int nq = ng + (ratio ? 1 : 0);
  std::vector<double> Vn, Qn;
  l2rows(REAL(V_), Vtot, D, Vn);
  l2rows(REAL(Q_), R, D, Qn);
  const int *klo = (op == 0) ? INTEGER(klo_) : nullptr, *khi = (op == 0) ? INTEGER(khi_) : nullptr;
  int nklo = (op == 0) ? Rf_length(klo_) : 0, nkhi = (op == 0) ? Rf_length(khi_) : 0;
  const double *thr = (op == 1) ? REAL(thr_) : nullptr;

  int nout = (op == 0) ? 4 : 1;
  SEXP out = PROTECT(Rf_allocVector(VECSXP, nout));
  std::vector<double *> optr(nout);
  for (int s = 0; s < nout; ++s) {
    SEXP m = (op == 2) ? Rf_allocMatrix(REALSXP, R, Vtot) : Rf_allocMatrix(REALSXP, nq, Vtot);
    SET_VECTOR_ELT(out, s, m); optr[s] = REAL(m);
  }
  int Vc = std::max(256, std::min(Vtot, (int)(16000000L / std::max(1, R))));
  std::vector<double> Sm((size_t)R * Vc);
  const double one = 1.0, zero = 0.0;
  for (int v0 = 0; v0 < Vtot; v0 += Vc) {
    int vc = std::min(Vc, Vtot - v0);
    // S (R x vc) = Qn (R x D) * Vn[v0:v0+vc, ]^T
    F77_CALL(dgemm)("N", "T", &R, &vc, &D, &one, Qn.data(), &R, Vn.data() + v0, &Vtot, &zero, Sm.data(), &R FCONE FCONE);
    if (op == 2) {
      for (int v = 0; v < vc; ++v) std::copy(Sm.begin() + (size_t)v * R, Sm.begin() + (size_t)(v + 1) * R, optr[0] + (size_t)(v0 + v) * R);
      continue;
    }
    parallel_for(vc, nt, [&](long a0, long a1) {
      std::vector<double> vals(nb);
      for (long v = a0; v < a1; ++v) {
        const double *col = Sm.data() + (size_t)v * R;
        long gv = v0 + v;
        for (int q = 0; q < nq; ++q) {
          for (int b = 0; b < nb; ++b)
            vals[b] = (q < ng) ? col[b * ng + q] : col[b * ng] / col[b * ng + 1];
          if (op == 1) {
            double t = thr[q + (size_t)nq * gv]; int cnt = 0;
            for (int b = 0; b < nb; ++b) { double x = (q < ng) ? vals[b] : std::fabs(vals[b] - 1.0); if (x >= t) cnt++; }
            optr[0][q + (size_t)nq * gv] = cnt;
            continue;
          }
          double m = 0; for (int b = 0; b < nb; ++b) m += vals[b]; m /= nb;
          double ss = 0; for (int b = 0; b < nb; ++b) ss += (vals[b] - m) * (vals[b] - m);
          optr[0][q + (size_t)nq * gv] = m;
          optr[1][q + (size_t)nq * gv] = nb > 1 ? std::sqrt(ss / (nb - 1)) : NA_REAL;
          int kl = klo[nklo == 1 ? 0 : gv], kh = khi[nkhi == 1 ? 0 : gv];
          if (kl >= 1 && kl <= nb) { std::nth_element(vals.begin(), vals.begin() + (kl - 1), vals.end()); optr[2][q + (size_t)nq * gv] = vals[kl - 1]; }
          else optr[2][q + (size_t)nq * gv] = NA_REAL;
          if (kh >= 1 && kh <= nb) { std::nth_element(vals.begin(), vals.begin() + (kh - 1), vals.end()); optr[3][q + (size_t)nq * gv] = vals[kh - 1]; }
          else optr[3][q + (size_t)nq * gv] = NA_REAL;
        }
      }
    });
  }
  UNPROTECT(1);
  return out;
}
