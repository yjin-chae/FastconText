# FastconText 0.1.0

* First release under the name FastconText (previously developed as conTextMetal).
* Built on conText 3.0.1 (Rodriguez, Spirling, Stewart, Avila). All functions, arguments
  and output objects of conText 3.0.1 are kept, and seeded results reproduce conText.
* New closed-form engine for `conText()`: one cluster-robust least-squares fit for all
  embedding dimensions, leave-one-out jackknife estimates from per-cluster summary
  statistics, and batched permutation tests.
* Bootstrap and permutation summaries in `get_nns()`, `get_cos_sim()`, `get_ncs()`,
  `get_nns_ratio()`, `contrast_nns()` and `bootstrap_nns()` are computed in one pass.
* Optional Apple GPU (Metal) engine on macOS; a multi-threaded double-precision CPU engine
  on Linux and macOS (and possibly Windows, but not tested!).
