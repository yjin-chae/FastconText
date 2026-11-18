\name{metal_info}
\alias{metal_info}
\alias{metal_available}
\title{Metal backend status and control}
\usage{
metal_info()
metal_available()
}
\value{
\code{metal_info()} returns a list with \code{available}, \code{device}, \code{max_buffer_bytes} and \code{reason}
(why no device is usable). \code{metal_available()} returns \code{TRUE} or \code{FALSE}.
}
\description{
Report whether an Apple GPU is usable from this R process. The backend is chosen with
\code{options(FastconText.backend = "auto" | "metal" | "cpu" | "original")}; see the package README.
Other options: \code{FastconText.jackknife} (\code{"auto"}, \code{"metal"}, \code{"cpu"}),
\code{FastconText.threads} and \code{FastconText.gpu_min_flops}.

\code{FastconText.jackknife_rank} controls jackknife leave-out fits that become rank-deficient (for example,
when a category has a single context or cluster). With \code{"conText"} (default) those fits are refit with
the original code, so results match conText 3.0.1. With \code{"na"} the standard errors and confidence
intervals of the coefficients involved in the collinearity are reported as \code{NA}, because their
leave-out estimates measure a different quantity. Both settings issue a warning naming the coefficients
and category sizes involved.
}
\examples{
metal_info()
}
