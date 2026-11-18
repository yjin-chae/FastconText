// FastconText Metal engine (compiled as Objective-C++ on macOS; stubs elsewhere).
//  * C_metal_info        : device discovery
//  * C_metal_gemm        : op(A) %*% op(B) in float32 via MPSMatrixMultiplication
//  * C_metal_spmm        : diag(scale) %*% CSR %*% E (sparse document-feature x embeddings)
//  * C_metal_sim_reduce  : batched cosine similarity (MPS GEMM) + per-column reductions on the GPU
//  * ctm_metal_jackknife : leave-one-cluster-out kernel in double-float (float2) arithmetic
#include "ctm.h"
#include <cmath>
#include <algorithm>
#include <thread>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#ifdef __APPLE__

// ------------------------------------------------------------------ MSL source
static const char *kMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---------- sparse (CSR) x dense ----------
kernel void spmm_csr(device const int   *rowptr [[buffer(0)]],
                     device const int   *colind [[buffer(1)]],
                     device const float *val    [[buffer(2)]],
                     device const float *E      [[buffer(3)]],
                     device const float *scale  [[buffer(4)]],
                     device float       *out    [[buffer(5)]],
                     constant uint2     &dims   [[buffer(6)]],   // (Dp, N)
                     uint2 gid [[thread_position_in_grid]]) {
  uint d = gid.x, r = gid.y, Dp = dims.x;
  if (d >= Dp || r >= dims.y) return;
  float acc = 0.0f;
  for (int k = rowptr[r]; k < rowptr[r + 1]; ++k) acc = fma(val[k], E[(uint)colind[k] * Dp + d], acc);
  out[r * Dp + d] = acc * scale[r];
}

// ---------- per-column reductions over bootstrap / permutation draws ----------
struct RedP { uint nb, ng, nq, Vc, v0, Vtot, op, nklo, nkhi; };

inline float sval(device const float *S, uint b, uint q, uint v, constant RedP &P) {
  uint row = b * P.ng;
  if (q < P.ng) return S[(row + q) * P.Vc + v];
  return S[row * P.Vc + v] / S[(row + 1) * P.Vc + v];
}
inline uint fkey(float f) { uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
inline float unkey(uint k) { uint u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k; return as_type<float>(u); }

// exact k-th smallest (1-based) by radix selection on order-preserving keys
inline float kth(device const float *S, uint q, uint v, constant RedP &P, uint k) {
  uint prefix = 0u;
  for (int bit = 31; bit >= 0; --bit) {
    uint maskhi = (bit == 31) ? 0u : (0xffffffffu << (uint)(bit + 1));
    uint cnt = 0u;
    for (uint b = 0; b < P.nb; ++b) {
      uint key = fkey(sval(S, b, q, v, P));
      if (((key ^ prefix) & maskhi) == 0u && ((key >> (uint)bit) & 1u) == 0u) cnt++;
    }
    if (k > cnt) { k -= cnt; prefix |= (1u << (uint)bit); }
  }
  return unkey(prefix);
}

kernel void colreduce(device const float *S   [[buffer(0)]],
                      constant RedP      &P   [[buffer(1)]],
                      device const int   *klo [[buffer(2)]],
                      device const int   *khi [[buffer(3)]],
                      device const float *thr [[buffer(4)]],
                      device float       *out [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
  uint v = gid.x, q = gid.y;
  if (v >= P.Vc || q >= P.nq) return;
  uint gv = P.v0 + v, o = q * P.Vc + v, plane = P.nq * P.Vc;
  float qnan = as_type<float>(0x7fc00000u);
  if (P.op == 1u) {
    float t = thr[q + P.nq * gv];
    uint cnt = 0u;
    for (uint b = 0; b < P.nb; ++b) {
      float x = sval(S, b, q, v, P);
      if (q >= P.ng) x = fabs(x - 1.0f);
      if (x >= t) cnt++;
    }
    out[o] = float(cnt);
    return;
  }
  float s = 0.0f, c = 0.0f;                     // Kahan-compensated mean
  for (uint b = 0; b < P.nb; ++b) { float y = sval(S, b, q, v, P) - c; float t = s + y; c = (t - s) - y; s = t; }
  float m = s / float(P.nb);
  float ss = 0.0f;
  for (uint b = 0; b < P.nb; ++b) { float dd = sval(S, b, q, v, P) - m; ss = fma(dd, dd, ss); }
  out[o] = m;
  out[plane + o] = (P.nb > 1u) ? sqrt(ss / float(P.nb - 1u)) : qnan;
  int kl = klo[P.nklo == 1u ? 0u : gv], kh = khi[P.nkhi == 1u ? 0u : gv];
  out[2u * plane + o] = (kl >= 1 && (uint)kl <= P.nb) ? kth(S, q, v, P, (uint)kl) : qnan;
  out[3u * plane + o] = (kh >= 1 && (uint)kh <= P.nb) ? kth(S, q, v, P, (uint)kh) : qnan;
}

// ---------- double-float (float2 = hi + lo) arithmetic ----------
inline float2 two_sum(float a, float b) { float s = a + b; float bb = s - a; float e = (a - (s - bb)) + (b - bb); return float2(s, e); }
inline float2 quick_two_sum(float a, float b) { float s = a + b; float e = b - (s - a); return float2(s, e); }
inline float2 two_prod(float a, float b) { float p = a * b; float e = fma(a, b, -p); return float2(p, e); }
inline float2 df_add(float2 a, float2 b) {
  float2 s = two_sum(a.x, b.x); float2 t = two_sum(a.y, b.y);
  s.y += t.x; s = quick_two_sum(s.x, s.y); s.y += t.y; return quick_two_sum(s.x, s.y);
}
inline float2 df_neg(float2 a) { return float2(-a.x, -a.y); }
inline float2 df_sub(float2 a, float2 b) { return df_add(a, df_neg(b)); }
inline float2 df_mul(float2 a, float2 b) {
  float2 p = two_prod(a.x, b.x); p.y = fma(a.x, b.y, p.y); p.y = fma(a.y, b.x, p.y); return quick_two_sum(p.x, p.y);
}
inline float2 df_fma(float2 a, float2 b, float2 c) { return df_add(df_mul(a, b), c); }
inline float2 df_div(float2 a, float2 b) {
  float q1 = a.x / b.x; float2 r = df_sub(a, df_mul(float2(q1, 0.0f), b));
  float q2 = r.x / b.x; r = df_sub(r, df_mul(float2(q2, 0.0f), b));
  float q3 = r.x / b.x; float2 q = quick_two_sum(q1, q2); return df_add(q, float2(q3, 0.0f));
}

#define PMAX 8
struct JKP { uint n, p, D, G; };

// Gauss-Jordan inverse with partial pivoting in double-float
inline bool df_invert(thread float2 *M, thread float2 *A, uint p) {
  for (uint i = 0; i < p * p; ++i) A[i] = float2(0.0f);
  for (uint i = 0; i < p; ++i) A[i * p + i] = float2(1.0f, 0.0f);
  float scale = 0.0f;
  for (uint i = 0; i < p; ++i) scale = max(scale, fabs(M[i * p + i].x));
  if (scale == 0.0f) return false;
  for (uint c = 0; c < p; ++c) {
    uint piv = c; float best = fabs(M[c * p + c].x);
    for (uint r = c + 1; r < p; ++r) if (fabs(M[r * p + c].x) > best) { best = fabs(M[r * p + c].x); piv = r; }
    if (best <= 1e-6f * scale) return false;
    if (piv != c) for (uint k = 0; k < p; ++k) {
      float2 t = M[c * p + k]; M[c * p + k] = M[piv * p + k]; M[piv * p + k] = t;
      t = A[c * p + k]; A[c * p + k] = A[piv * p + k]; A[piv * p + k] = t;
    }
    float2 inv = df_div(float2(1.0f, 0.0f), M[c * p + c]);
    for (uint k = 0; k < p; ++k) { M[c * p + k] = df_mul(M[c * p + k], inv); A[c * p + k] = df_mul(A[c * p + k], inv); }
    for (uint r = 0; r < p; ++r) if (r != c) {
      float2 f = M[r * p + c];
      if (f.x == 0.0f && f.y == 0.0f) continue;
      for (uint k = 0; k < p; ++k) {
        M[r * p + k] = df_sub(M[r * p + k], df_mul(f, M[c * p + k]));
        A[r * p + k] = df_sub(A[r * p + k], df_mul(f, A[c * p + k]));
      }
    }
  }
  return true;
}

kernel void jk_df64(device const float2 *B   [[buffer(0)]],   // D*p
                    device const float2 *H   [[buffer(1)]],   // p*p
                    device const float2 *Hg  [[buffer(2)]],   // G*p*p
                    device const float2 *Zg  [[buffer(3)]],   // G*D*p
                    device const float2 *M1  [[buffer(4)]],   // D*p*p
                    device const float2 *M2  [[buffer(5)]],   // D*p^3
                    device const float2 *M4  [[buffer(6)]],   // p^4
                    device const uint   *ng  [[buffer(7)]],
                    constant JKP        &P   [[buffer(8)]],
                    device float2       *out [[buffer(9)]],   // (p-1)*G, out[(k-1)*G + j]
                    device const uchar  *skip [[buffer(10)]], // rank-deficient leave-outs (done on CPU)
                    device uchar        *redo [[buffer(11)]], // degenerate variance -> recompute on CPU
                    device const float  *tss  [[buffer(12)]], // D, variance reference scale
                    uint j [[thread_position_in_grid]]) {
  if (j >= P.G || skip[j]) return;
  const uint p = P.p, D = P.D, G = P.G, p2 = p * p;
  float2 Hj[PMAX * PMAX], Aj[PMAX * PMAX], aa[PMAX * PMAX], K[PMAX * PMAX], u[PMAX], delta[PMAX], P2[PMAX];
  device const float2 *hj = Hg + (ulong)j * p2;
  for (uint i = 0; i < p2; ++i) Hj[i] = df_sub(H[i], hj[i]);
  if (!df_invert(Hj, Aj, p)) {
    for (uint k = 1; k < p; ++k) out[(ulong)(k - 1) * G + j] = float2(as_type<float>(0x7fc00000u), 0.0f);
    return;
  }
  const float2 one = float2(1.0f, 0.0f);
  float2 Gj = float2(float(G - 1u), 0.0f), Nj = float2(float(P.n - ng[j]), 0.0f);
  float2 cj = df_mul(df_div(Gj, df_sub(Gj, one)), df_div(df_sub(Nj, one), df_sub(Nj, float2(float(p), 0.0f))));
  for (uint k = 1; k < p; ++k) {
    thread const float2 *a = Aj + k * p;
    for (uint x = 0; x < p; ++x) for (uint y = 0; y < p; ++y) aa[x * p + y] = df_mul(a[x], a[y]);
    for (uint c = 0; c < p; ++c) for (uint c2 = 0; c2 < p; ++c2) {
      float2 s = float2(0.0f);
      for (uint b = 0; b < p; ++b) for (uint b2 = 0; b2 < p; ++b2) s = df_fma(aa[b * p + b2], M4[(b * p + c) * p2 + b2 * p + c2], s);
      K[c * p + c2] = s;
    }
    for (uint c = 0; c < p; ++c) { float2 s = float2(0.0f); for (uint b = 0; b < p; ++b) s = df_fma(a[b], hj[b * p + c], s); u[c] = s; }
    float2 acc = float2(0.0f);
    for (uint d = 0; d < D; ++d) {
      device const float2 *zj = Zg + ((ulong)j * D + d) * p;
      for (uint x = 0; x < p; ++x) { float2 s = float2(0.0f); for (uint y = 0; y < p; ++y) s = df_sub(s, df_mul(Aj[x * p + y], zj[y])); delta[x] = s; }
      device const float2 *m1 = M1 + (ulong)d * p2;
      device const float2 *m2 = M2 + (ulong)d * p2 * p;
      float2 t1 = float2(0.0f);
      for (uint v = 0; v < p2; ++v) t1 = df_fma(aa[v], m1[v], t1);
      for (uint c = 0; c < p; ++c) P2[c] = float2(0.0f);
      for (uint ab = 0; ab < p2; ++ab) { float2 f = aa[ab]; for (uint c = 0; c < p; ++c) P2[c] = df_fma(f, m2[ab * p + c], P2[c]); }
      float2 t2 = float2(0.0f);
      for (uint c = 0; c < p; ++c) t2 = df_fma(P2[c], delta[c], t2);
      float2 t3 = float2(0.0f);
      for (uint x = 0; x < p; ++x) { float2 s = float2(0.0f); for (uint y = 0; y < p; ++y) s = df_fma(K[x * p + y], delta[y], s); t3 = df_fma(delta[x], s, t3); }
      float2 az = float2(0.0f), ud = float2(0.0f);
      for (uint x = 0; x < p; ++x) { az = df_fma(a[x], zj[x], az); ud = df_fma(u[x], delta[x], ud); }
      float2 r = df_sub(az, ud);
      float2 inner = df_sub(df_add(df_sub(t1, df_add(t2, t2)), t3), df_mul(r, r));
      float2 se2 = df_mul(cj, inner);
      float ref = tss[d] / (float(P.n - ng[j]) - float(p)) * Aj[k * p + k].x;
      float mag = fabs(t1.x) + 2.0f * fabs(t2.x) + fabs(t3.x) + fabs(df_mul(r, r).x);
      if (fabs(se2.x) <= 1e-12f * ref || se2.x < 0.0f || fabs(inner.x) < 1e-6f * mag) redo[j] = (uchar)1;
      float2 bn = df_add(B[(ulong)d * p + k], delta[k]);
      acc = df_add(acc, df_sub(df_mul(bn, bn), se2));
    }
    out[(ulong)(k - 1) * G + j] = acc;
  }
}
)MSL";

// ------------------------------------------------------------------ context
struct MetalCtx {
  bool tried = false, ok = false;
  std::string name, err;
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;
  id<MTLComputePipelineState> spmm = nil, reduce = nil, jk = nil;
};
static MetalCtx &ctx() { static MetalCtx c; return c; }

static id<MTLComputePipelineState> make_pso(MetalCtx &C, const char *fn) {
  NSError *e = nil;
  id<MTLFunction> f = [C.lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
  if (!f) { C.err = std::string("missing kernel ") + fn; return nil; }
  id<MTLComputePipelineState> p = [C.dev newComputePipelineStateWithFunction:f error:&e];
  if (!p) C.err = std::string("pipeline ") + fn + ": " + (e ? [[e localizedDescription] UTF8String] : "?");
  return p;
}

static bool metal_init() {
  MetalCtx &C = ctx();
  if (C.tried) return C.ok;
  C.tried = true;
  @autoreleasepool {
    C.dev = MTLCreateSystemDefaultDevice();
    if (!C.dev) { C.err = "no Metal device visible to this process"; return false; }
    C.name = [[C.dev name] UTF8String];
    if (!MPSSupportsMTLDevice(C.dev)) { C.err = "Metal Performance Shaders not supported on " + C.name; return false; }
    C.queue = [C.dev newCommandQueue];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) { opt.mathMode = MTLMathModeSafe; }
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError *e = nil;
    C.lib = [C.dev newLibraryWithSource:[NSString stringWithUTF8String:kMSL] options:opt error:&e];
    if (!C.lib) { C.err = std::string("MSL compile failed: ") + (e ? [[e localizedDescription] UTF8String] : "?"); return false; }
    C.spmm = make_pso(C, "spmm_csr");
    C.reduce = make_pso(C, "colreduce");
    C.jk = make_pso(C, "jk_df64");
    if (!C.spmm || !C.reduce || !C.jk) return false;
  }
  C.ok = true;
  return true;
}

bool ctm_metal_available(std::string &name) {
  bool ok = metal_init();
  name = ok ? ctx().name : ctx().err;
  return ok;
}

static id<MTLBuffer> newbuf(size_t bytes) {
  return [ctx().dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
}
static size_t pad4(size_t x) { return (x + 3) / 4 * 4; }

// R col-major (nr x nc) -> row-major float (nr x ldc), optional row L2 normalisation (double precision)
static void to_rowmajor(const double *X, size_t nr, size_t nc, size_t ldc, float *dst, bool normalize) {
  int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  auto work = [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      double s = 1.0;
      if (normalize) { double ss = 0; for (size_t c = 0; c < nc; ++c) { double v = X[c * nr + r]; ss += v * v; } s = 1.0 / std::sqrt(ss); }
      float *row = dst + r * ldc;
      for (size_t c = 0; c < nc; ++c) row[c] = (float)(X[c * nr + r] * s);
      for (size_t c = nc; c < ldc; ++c) row[c] = 0.0f;
    }
  };
  if (nr < 4096) { work(0, nr); return; }
  std::vector<std::thread> th; size_t ch = (nr + nt - 1) / nt;
  for (int t = 0; t < nt; ++t) { size_t b = t * ch, e = std::min(nr, b + ch); if (b < e) th.emplace_back(work, b, e); }
  for (auto &x : th) x.join();
}

static MPSMatrix *mps_matrix(id<MTLBuffer> buf, size_t offset, size_t rows, size_t cols, size_t ld) {
  MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:ld * sizeof(float) dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:buf offset:offset descriptor:d];
}

static std::string cb_error(id<MTLCommandBuffer> cb) {
  if (cb.status == MTLCommandBufferStatusError) return std::string("GPU command failed: ") + (cb.error ? [[cb.error localizedDescription] UTF8String] : "?");
  return "";
}

// ------------------------------------------------------------------ .Call entry points
extern "C" SEXP C_metal_info() {
  std::string name;
  bool ok = ctm_metal_available(name);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(out, 0, Rf_ScalarLogical(ok));
  SET_VECTOR_ELT(out, 1, Rf_mkString(ok ? name.c_str() : ""));
  SET_VECTOR_ELT(out, 2, Rf_ScalarReal(ok ? (double)[ctx().dev maxBufferLength] : NA_REAL));
  SET_VECTOR_ELT(out, 3, Rf_mkString(ok ? "" : name.c_str()));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 4));
  const char *n[] = {"available", "device", "max_buffer_bytes", "reason"};
  for (int i = 0; i < 4; ++i) SET_STRING_ELT(nm, i, Rf_mkChar(n[i]));
  Rf_setAttrib(out, R_NamesSymbol, nm);
  UNPROTECT(2);
  return out;
}

// C = op(A) %*% op(B)   (float32 on GPU, returned as double)
extern "C" SEXP C_metal_gemm(SEXP A_, SEXP B_, SEXP ta_, SEXP tb_) {
  std::string err;
  if (!metal_init()) Rf_error("Metal unavailable: %s", ctx().err.c_str());
  bool ta = Rf_asLogical(ta_), tb = Rf_asLogical(tb_);
  size_t rA = Rf_nrows(A_), cA = Rf_ncols(A_), rB = Rf_nrows(B_), cB = Rf_ncols(B_);
  size_t m = ta ? cA : rA, kA = ta ? rA : cA, kB = tb ? cB : rB, n = tb ? rB : cB;
  if (kA != kB) Rf_error("non-conformable matrices");
  SEXP Cm = PROTECT(Rf_allocMatrix(REALSXP, (int)m, (int)n));
  @autoreleasepool {
    MetalCtx &C = ctx();
    // row-major views: A_rm = t(A) (cA x rA), B_rm = t(B) (cB x rB); C_rm = t(C) = op(B)^T op(A)^T
    size_t ldA = pad4(rA), ldB = pad4(rB), ldC = pad4(m);
    id<MTLBuffer> bA = newbuf(cA * ldA * 4), bB = newbuf(cB * ldB * 4), bC = newbuf(n * ldC * 4);
    if (!bA || !bB || !bC) err = "buffer allocation failed";
    else {
      // memory of col-major X (r x c) is row-major (c x r): copy columns as rows
      const double *a = REAL(A_), *b = REAL(B_);
      float *fa = (float *)bA.contents, *fb = (float *)bB.contents;
      for (size_t c = 0; c < cA; ++c) { for (size_t r = 0; r < rA; ++r) fa[c * ldA + r] = (float)a[c * rA + r]; for (size_t r = rA; r < ldA; ++r) fa[c * ldA + r] = 0; }
      for (size_t c = 0; c < cB; ++c) { for (size_t r = 0; r < rB; ++r) fb[c * ldB + r] = (float)b[c * rB + r]; for (size_t r = rB; r < ldB; ++r) fb[c * ldB + r] = 0; }
      MPSMatrix *L = mps_matrix(bB, 0, cB, rB, ldB), *R = mps_matrix(bA, 0, cA, rA, ldA), *Res = mps_matrix(bC, 0, n, m, ldC);
      MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:C.dev transposeLeft:tb transposeRight:ta
                                                                         resultRows:n resultColumns:m interiorColumns:kA alpha:1.0 beta:0.0];
      id<MTLCommandBuffer> cb = [C.queue commandBuffer];
      [mm encodeToCommandBuffer:cb leftMatrix:L rightMatrix:R resultMatrix:Res];
      [cb commit]; [cb waitUntilCompleted];
      err = cb_error(cb);
      if (err.empty()) {
        const float *fc = (const float *)bC.contents; double *co = REAL(Cm);
        for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < m; ++i) co[j * m + i] = fc[j * ldC + i];
      }
    }
  }
  if (!err.empty()) { UNPROTECT(1); Rf_error("%s", err.c_str()); }
  UNPROTECT(1);
  return Cm;
}

// out (N x D) = diag(scale) %*% M %*% E, with M given in CSR form (0-based), E (V x D)
extern "C" SEXP C_metal_spmm(SEXP rowptr_, SEXP colind_, SEXP val_, SEXP E_, SEXP scale_) {
  std::string err;
  if (!metal_init()) Rf_error("Metal unavailable: %s", ctx().err.c_str());
  size_t N = Rf_length(scale_), V = Rf_nrows(E_), D = Rf_ncols(E_), nnz = Rf_length(colind_);
  SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)N, (int)D));
  @autoreleasepool {
    MetalCtx &C = ctx();
    size_t Dp = pad4(D);
    id<MTLBuffer> bRp = newbuf((N + 1) * 4), bCi = newbuf(nnz * 4), bV = newbuf(nnz * 4), bE = newbuf(V * Dp * 4),
                  bS = newbuf(N * 4), bO = newbuf(N * Dp * 4), bD = newbuf(8);
    std::copy(INTEGER(rowptr_), INTEGER(rowptr_) + N + 1, (int *)bRp.contents);
    std::copy(INTEGER(colind_), INTEGER(colind_) + nnz, (int *)bCi.contents);
    { float *f = (float *)bV.contents; const double *x = REAL(val_); for (size_t i = 0; i < nnz; ++i) f[i] = (float)x[i]; }
    { float *f = (float *)bS.contents; const double *x = REAL(scale_); for (size_t i = 0; i < N; ++i) f[i] = (float)x[i]; }
    to_rowmajor(REAL(E_), V, D, Dp, (float *)bE.contents, false);
    uint32_t dims[2] = {(uint32_t)Dp, (uint32_t)N}; memcpy(bD.contents, dims, 8);
    id<MTLCommandBuffer> cb = [C.queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:C.spmm];
    [en setBuffer:bRp offset:0 atIndex:0]; [en setBuffer:bCi offset:0 atIndex:1]; [en setBuffer:bV offset:0 atIndex:2];
    [en setBuffer:bE offset:0 atIndex:3]; [en setBuffer:bS offset:0 atIndex:4]; [en setBuffer:bO offset:0 atIndex:5];
    [en setBuffer:bD offset:0 atIndex:6];
    [en dispatchThreads:MTLSizeMake(Dp, N, 1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(Dp, 64), 4, 1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    err = cb_error(cb);
    if (err.empty()) {
      const float *o = (const float *)bO.contents; double *r = REAL(out);
      for (size_t i = 0; i < N; ++i) for (size_t d = 0; d < D; ++d) r[d * N + i] = o[i * Dp + d];
    }
  }
  if (!err.empty()) { UNPROTECT(1); Rf_error("%s", err.c_str()); }
  UNPROTECT(1);
  return out;
}

// Same contract as C_sim_reduce_cpu (see ctm_cpu.cpp).
extern "C" SEXP C_metal_sim_reduce(SEXP V_, SEXP Q_, SEXP nb_, SEXP ng_, SEXP ratio_, SEXP op_,
                                   SEXP klo_, SEXP khi_, SEXP thr_) {
  std::string err;
  if (!metal_init()) Rf_error("Metal unavailable: %s", ctx().err.c_str());
  size_t Vtot = Rf_nrows(V_), D = Rf_ncols(V_), R = Rf_nrows(Q_);
  uint32_t nb = Rf_asInteger(nb_), ng = Rf_asInteger(ng_), op = Rf_asInteger(op_);
  bool ratio = Rf_asLogical(ratio_);
  if ((size_t)Rf_ncols(Q_) != D || R != (size_t)nb * ng) Rf_error("dimension mismatch in sim_reduce");
  uint32_t nq = ng + (ratio ? 1 : 0);
  int nout = (op == 0) ? 4 : 1;
  SEXP out = PROTECT(Rf_allocVector(VECSXP, nout));
  std::vector<double *> optr(nout);
  for (int s = 0; s < nout; ++s) {
    SEXP m = (op == 2) ? Rf_allocMatrix(REALSXP, (int)R, (int)Vtot) : Rf_allocMatrix(REALSXP, (int)nq, (int)Vtot);
    SET_VECTOR_ELT(out, s, m); optr[s] = REAL(m);
  }
  @autoreleasepool {
    MetalCtx &C = ctx();
    size_t Dp = pad4(D);
    id<MTLBuffer> bV = newbuf(Vtot * Dp * 4), bQ = newbuf(R * Dp * 4);
    to_rowmajor(REAL(V_), Vtot, D, Dp, (float *)bV.contents, true);
    to_rowmajor(REAL(Q_), R, D, Dp, (float *)bQ.contents, true);
    size_t nklo = 1, nkhi = 1;
    id<MTLBuffer> bKlo = newbuf(4), bKhi = newbuf(4), bThr = newbuf(4);
    if (op == 0) {
      nklo = Rf_length(klo_); nkhi = Rf_length(khi_);
      bKlo = newbuf(nklo * 4); bKhi = newbuf(nkhi * 4);
      std::copy(INTEGER(klo_), INTEGER(klo_) + nklo, (int *)bKlo.contents);
      std::copy(INTEGER(khi_), INTEGER(khi_) + nkhi, (int *)bKhi.contents);
    } else if (op == 1) {
      bThr = newbuf((size_t)nq * Vtot * 4);
      float *t = (float *)bThr.contents; const double *x = REAL(thr_);
      for (size_t i = 0; i < (size_t)nq * Vtot; ++i) t[i] = (float)x[i];
    }
    size_t Vc = std::min(Vtot, std::max<size_t>(1024, (size_t)(64u << 20) / std::max<size_t>(R, 1)));
    id<MTLBuffer> bS = newbuf(R * Vc * 4), bO = newbuf((size_t)4 * nq * Vc * 4), bP = newbuf(sizeof(uint32_t) * 9);
    MPSMatrix *Ql = mps_matrix(bQ, 0, R, D, Dp);
    for (size_t v0 = 0; v0 < Vtot && err.empty(); v0 += Vc) {
      size_t vc = std::min(Vc, Vtot - v0);
      MPSMatrix *Vr = mps_matrix(bV, v0 * Dp * 4, vc, D, Dp), *Sm = mps_matrix(bS, 0, R, vc, vc);
      MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:C.dev transposeLeft:NO transposeRight:YES
                                                                         resultRows:R resultColumns:vc interiorColumns:D alpha:1.0 beta:0.0];
      id<MTLCommandBuffer> cb = [C.queue commandBuffer];
      [mm encodeToCommandBuffer:cb leftMatrix:Ql rightMatrix:Vr resultMatrix:Sm];
      if (op != 2) {
        uint32_t P[9] = {nb, ng, nq, (uint32_t)vc, (uint32_t)v0, (uint32_t)Vtot, op, (uint32_t)nklo, (uint32_t)nkhi};
        memcpy(bP.contents, P, sizeof(P));
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:C.reduce];
        [en setBuffer:bS offset:0 atIndex:0]; [en setBuffer:bP offset:0 atIndex:1];
        [en setBuffer:bKlo offset:0 atIndex:2]; [en setBuffer:bKhi offset:0 atIndex:3];
        [en setBuffer:bThr offset:0 atIndex:4]; [en setBuffer:bO offset:0 atIndex:5];
        [en dispatchThreads:MTLSizeMake(vc, nq, 1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(vc, 256), 1, 1)];
        [en endEncoding];
      }
      [cb commit]; [cb waitUntilCompleted];
      err = cb_error(cb);
      if (!err.empty()) break;
      if (op == 2) {
        const float *s = (const float *)bS.contents;
        for (size_t r = 0; r < R; ++r) for (size_t v = 0; v < vc; ++v) optr[0][r + R * (v0 + v)] = s[r * vc + v];
      } else {
        const float *o = (const float *)bO.contents;
        size_t plane = (size_t)nq * vc;
        for (int s = 0; s < nout; ++s) for (size_t q = 0; q < nq; ++q) for (size_t v = 0; v < vc; ++v) {
          float x = o[s * plane + q * vc + v];
          optr[s][q + nq * (v0 + v)] = std::isnan(x) ? NA_REAL : (double)x;
        }
      }
    }
  }
  if (!err.empty()) { UNPROTECT(1); Rf_error("%s", err.c_str()); }
  UNPROTECT(1);
  return out;
}

// ------------------------------------------------------------------ jackknife kernel driver
static id<MTLBuffer> df_buffer(const std::vector<double> &x) {
  id<MTLBuffer> b = newbuf(x.size() * 8);
  float *f = (float *)b.contents;
  for (size_t i = 0; i < x.size(); ++i) { float hi = (float)x[i]; f[2 * i] = hi; f[2 * i + 1] = (float)(x[i] - (double)hi); }
  return b;
}

bool ctm_metal_jackknife(const JKStats &S, const std::vector<unsigned char> &skip, double *out,
                         std::vector<unsigned char> &redo, std::string &err) {
  if (!metal_init()) { err = ctx().err; return false; }
  if (S.p > 8) { err = "more than 8 regression terms"; return false; }
  if (S.n >= (1 << 24)) { err = "too many observations for float counters"; return false; }
  @autoreleasepool {
    MetalCtx &C = ctx();
    id<MTLBuffer> bB = df_buffer(S.B), bH = df_buffer(S.H), bHg = df_buffer(S.Hg), bZg = df_buffer(S.Zg),
                  bM1 = df_buffer(S.M1), bM2 = df_buffer(S.M2), bM4 = df_buffer(S.M4);
    id<MTLBuffer> bng = newbuf(S.G * 4);
    { uint32_t *u = (uint32_t *)bng.contents; for (int g = 0; g < S.G; ++g) u[g] = (uint32_t)S.ng[g]; }
    uint32_t P[4] = {(uint32_t)S.n, (uint32_t)S.p, (uint32_t)S.D, (uint32_t)S.G};
    id<MTLBuffer> bP = newbuf(sizeof(P)); memcpy(bP.contents, P, sizeof(P));
    id<MTLBuffer> bO = newbuf((size_t)(S.p - 1) * S.G * 8);
    id<MTLBuffer> bSkip = newbuf(S.G), bRedo = newbuf(S.G), bTss = newbuf((size_t)S.D * 4);
    memcpy(bSkip.contents, skip.data(), S.G);
    memset(bRedo.contents, 0, S.G);
    { float *t = (float *)bTss.contents; for (int d = 0; d < S.D; ++d) t[d] = (float)S.tss[d]; }
    if (!bZg || !bO) { err = "buffer allocation failed"; return false; }
    id<MTLCommandBuffer> cb = [C.queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:C.jk];
    id<MTLBuffer> bufs[13] = {bB, bH, bHg, bZg, bM1, bM2, bM4, bng, bP, bO, bSkip, bRedo, bTss};
    for (int i = 0; i < 13; ++i) [en setBuffer:bufs[i] offset:0 atIndex:i];
    NSUInteger tg = std::min<NSUInteger>(64, C.jk.maxTotalThreadsPerThreadgroup);
    [en dispatchThreads:MTLSizeMake(S.G, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    err = cb_error(cb);
    if (!err.empty()) return false;
    redo.assign((const unsigned char *)bRedo.contents, (const unsigned char *)bRedo.contents + S.G);
    const float *o = (const float *)bO.contents;
    for (size_t i = 0; i < (size_t)(S.p - 1) * S.G; ++i) {
      float hi = o[2 * i], lo = o[2 * i + 1];
      out[i] = std::isnan(hi) ? NA_REAL : (double)hi + (double)lo;
    }
  }
  return true;
}

#else  // ---------------------------------------------------------------- non-Apple stubs

bool ctm_metal_available(std::string &name) { name = "Metal requires macOS"; return false; }
bool ctm_metal_jackknife(const JKStats &, const std::vector<unsigned char> &, double *, std::vector<unsigned char> &, std::string &err) { err = "Metal requires macOS"; return false; }
extern "C" SEXP C_metal_info() {
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(out, 0, Rf_ScalarLogical(0)); SET_VECTOR_ELT(out, 1, Rf_mkString(""));
  SET_VECTOR_ELT(out, 2, Rf_ScalarReal(NA_REAL)); SET_VECTOR_ELT(out, 3, Rf_mkString("Metal requires macOS"));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 4));
  const char *n[] = {"available", "device", "max_buffer_bytes", "reason"};
  for (int i = 0; i < 4; ++i) SET_STRING_ELT(nm, i, Rf_mkChar(n[i]));
  Rf_setAttrib(out, R_NamesSymbol, nm);
  UNPROTECT(2); return out;
}
extern "C" SEXP C_metal_gemm(SEXP, SEXP, SEXP, SEXP) { Rf_error("Metal requires macOS"); return R_NilValue; }
extern "C" SEXP C_metal_spmm(SEXP, SEXP, SEXP, SEXP, SEXP) { Rf_error("Metal requires macOS"); return R_NilValue; }
extern "C" SEXP C_metal_sim_reduce(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP) { Rf_error("Metal requires macOS"); return R_NilValue; }

#endif
