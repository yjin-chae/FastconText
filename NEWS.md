# FastconText 0.2.0

* `conText()` accepts `lm()`-style formulas on the right-hand side, including interactions
  (`party * gender`, `party:gender`, `(party + gender + region)^2`, `gender / party`), terms and function calls such as `log(x)`, `I(x^2)`, and `poly(x, 2)`
* New `na.rm` argument in `conText()` and new option `FastconText.na_action` (`"conText"` or
  `"drop"`). By default, contexts with missing covariate values are retained and run with the original conText code. This slows down the estimation.
  With `na.rm = TRUE` or `options(FastconText.na_action = "drop")`, they are dropped before the
  regression and jackknife and permutations (listwise deletion, as in `lm()`).

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
