// FastconText: shared declarations for the CPU (double) and Metal (float32 / double-float) engines.
#ifndef CTM_H
#define CTM_H

#include <vector>
#include <string>
#include <cstddef>

// Sufficient statistics for the closed-form leave-one-cluster-out jackknife.
// All arrays are double precision and use the layouts documented below.
struct JKStats {
  int n = 0, p = 0, D = 0, G = 0;
  std::vector<double> B;    // D x p        B[d*p + a]          full-sample coefficients
  std::vector<double> H;    // p x p        H[a*p + b]          X'WX
  std::vector<double> Hg;   // G x p x p    Hg[(g*p + a)*p + b] per-cluster X_g'W_gX_g
  std::vector<double> Zg;   // G x D x p    Zg[(g*D + d)*p + a] per-cluster scores X_g'W_g e_g
  std::vector<double> M1;   // D x p x p    sum_g z_gd z_gd'
  std::vector<double> M2;   // D x p^3      sum_g z_gd[a] Hg[b][c]
  std::vector<double> M4;   // p^4          sum_g Hg[b][c] Hg[b2][c2]
  std::vector<int> ng;      // G            observations per cluster
  std::vector<double> tss;  // D            weighted total sum of squares of each outcome (variance reference)
};

// estimatr 2.0.1 conventions reproduced by the closed-form engines
constexpr double CTM_RANK_TOL = 1e-7;   // order-preserving rank test on unit-norm columns (dqrdc2 rule)
constexpr double CTM_ZERO_TOL = 1e-12;
constexpr double CTM_CANCEL_TOL = 1e-6; // closed-form terms cancelling beyond this -> direct score sum   // variance <= tol * reference is numerically zero -> SE is NA

// rank detection on a Gram matrix (row-major p x p); keep[c] = 1 for kept columns. Returns rank.
int ctm_rank_keep(const double *H, int p, int *keep);

// Metal engine (defined in ctm_metal.cpp; returns false + message when unavailable).
bool ctm_metal_available(std::string &name);
bool ctm_metal_jackknife(const JKStats &S, const std::vector<unsigned char> &skip, double *out,
                         std::vector<unsigned char> &redo, std::string &err);

#endif
