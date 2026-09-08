packages <- c(
  "dplyr",
  "ggplot2",
  "kernlab",
  "readr",
  "scales",
  "scatterplot3d",
  "tidyr"
)

missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}
