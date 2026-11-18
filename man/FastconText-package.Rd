% Please edit documentation in R/conText-package.R
\docType{package}
\name{FastconText-package}
\alias{FastconText}
\alias{FastconText-package}
\title{FastconText: Fast Embedding Regression for 'a la Carte' on Text (ConText) Models}
\description{
A drop-in, faster implementation of the 'conText' package (version 3.0.1) by Rodriguez,
Spirling, Stewart, and Avila. Functions, arguments, output objects and random-number streams
are the same, so seeded results reproduce 'conText'. The speedup comes from closed-form
estimators in multi-threaded double-precision C++, with an optional Apple GPU
(Metal) engine on macOS. Methods: Khodak et al. (2018) \doi{10.48550/arXiv.1805.05388};
Rodriguez et al. (2023) \doi{10.1017/S0003055422001228}; Green et al. (2025) \doi{10.1017/pan.2024.22}.
}
\seealso{
\url{https://github.com/yjin-chae/FastconText}
}
\author{
\strong{Maintainer}: Youngjin YJ Chae \email{yj.chae@rutgers.edu} (\href{https://orcid.org/0000-0001-5074-9712}{ORCID})

The original 'conText' package, on which this package is built, was written by Pedro L. Rodriguez,
Arthur Spirling, Brandon Stewart and Sofia Avila, with contributions from Christopher Barrie
(\url{https://github.com/prodriguezsosa/conText}). FastconText is not affiliated with or
endorsed by the 'conText' authors.
}
\keyword{internal}
