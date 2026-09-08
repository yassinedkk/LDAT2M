required_packages <- c(
  "R2WinBUGS",
  "coda",
  "rjags",
  "ggplot2",
  "HDInterval"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}
