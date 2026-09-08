library(ggplot2)
library(tidyr)
library(dplyr)
library(readr)

winequality.red= read.csv("data/winequality-red.csv", sep = ";")
wine_quality <- winequality.red$quality[1:500]
wine_features <- winequality.red[1:500,1:11]
str(wine_features)


wine_long <- winequality.red %>%
  pivot_longer(
    cols = -quality,
    names_to = "variable",
    values_to = "value"
  )

# Création des boxplots

ggplot(wine_long, aes(x = factor(quality), y = value, fill = factor(quality))) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +  # Change ncol selon l'espace dispo
  labs(
    title = "Distribution des variables explicatives par qualité du vin",
    x = "Qualité du vin",
    y = "Valeur"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 12),
    axis.text.x = element_text(angle = 0)
  )
#PCA############################################################################
wine_quality <- winequality.red$quality[1:500]
wine_features <- winequality.red[1:500,1:11]
#centrage les données 
z <- sapply(wine_features, function(x) scale(x, scale=FALSE))

n <- nrow(z)
p <- ncol(z)
# 2. Calcul de la matrice de covariance
Sigma <- (t(z) %*% z) /n
# 3. Décomposition en valeurs propres

eig <- eigen(Sigma)

eigenvalues <- eig$values
eigenvectors <- eig$vectors
# Calcul de la fraction de variance totale expliquée
fraction=cumsum(eigenvalues)/sum(eigenvalues)

# Trouver le plus petit r tel que f(r) ≥ α
r=which.max(fraction >=0.97)

eigenvector_red=eigenvectors[,1:r]
# Projection
A=z %*% eigenvector_red 

variance_df <- data.frame(
  Composante = 1:length(eigenvalues),
  Individual_var = eigenvalues/sum(eigenvalues),
  fraction = fraction
)
ggplot(variance_df, aes(x = Composante, y = fraction)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.97, linetype = "dashed", color = "red") +
  labs(title = "Cumulative Variance Explained by Principal Components",
       y = "Cumulative Variance Explained",
       x = "Principal Component") +
  theme_minimal()

pca_df <- data.frame(PC1 = A[,1], PC2 = A[,2], Quality =wine_quality)
ggplot(pca_df, aes(x = PC1, y = PC2, color =factor(Quality) )) +
  geom_point() +scale_color_manual(
    values = c("3" = "red", "4" = "blue", "5" = "green", "6" = "purple", "7" = "orange", "8" = "pink"))+
  labs(title = "PCA of Wine Quality Dataset",
       x = "First Principal Component",
       y = "Second Principal Component") +
  theme_minimal()

#kernel pca#################################################################################
library(kernlab)

#Calcul de la matrice de kernel 
K <- kernelMatrix(rbfdot(sigma = 0.0001), as.matrix(wine_features))

# Centrage pour Kernel PCA 
n <- nrow(K)
H <- diag(n) - matrix(1/n, n, n)
K_centered <- H %*% K %*% H

eig_ker=eigen(K_centered)

eigenvalues_ker <- eig_ker$values
eigenvectors_ker <- eig_ker$vectors
var=eigenvalues_ker/n

for(i in 1:ncol(eigenvectors_ker )){
  eigenvectors_ker[,i]=eigenvectors_ker[,i]/sqrt(eigenvalues_ker[i])
}

fraction_ker=cumsum(var)/sum(var)
r_ker=which.max(fraction_ker >=0.97)

eigenvector_ker_red=eigenvectors_ker[,1:r_ker]
# Projection
A_ker=K_centered %*% eigenvector_ker_red 

variance_df_ker <- data.frame(
  Composante = 1:length(eigenvalues_ker),
  Individual_var = eigenvalues_ker/sum(eigenvalues_ker),
  fraction = fraction_ker
)

ggplot(variance_df_ker, aes(x = Composante, y = fraction)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.97, linetype = "dashed", color = "red") +
  labs(title = "Cumulative Variance Explained by Principal Components",
       y = "Cumulative Variance Explained",
       x = "Principal Component") +
  theme_minimal()
############################
library(scatterplot3d)
pca_df_ker <- data.frame(
  PC1 = A_ker[,1],
  PC2 = A_ker[,2],
  PC3 = A_ker[,3],
  Quality = as.factor(wine_quality)
)

cols <- c("3" = "red", "4" = "blue", "5" = "green", 
          "6" = "purple", "7" = "orange", "8" = "pink")[pca_df_ker$Quality]

scatterplot3d(
  x = pca_df_ker$PC1,
  y = pca_df_ker$PC2,
  z = pca_df_ker$PC3,
  color = cols,
  pch = 19,
  xlab = "PC1",
  ylab = "PC2",
  zlab = "PC3",
  main = "Projection 3D des 3 premières composantes (KPCA)"
)
legend("topright", legend = levels(pca_df_ker$Quality), 
       col = c("red", "blue", "green", "purple", "orange", "pink"), pch = 19)
############################
pca_df_ker <- data.frame(
  PC1 = A_ker[,1], 
  PC2 = A_ker[,2], 
  PC3 = A_ker[,3], 
  Quality = factor(wine_quality)
)


pca_df_ker$PC3_size <- scales::rescale(abs(pca_df_ker$PC3), to = c(1, 5))


ggplot(pca_df_ker, aes(x = PC1, y = PC2, color = Quality, size = PC3_size)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = c("3" = "red", "4" = "blue", "5" = "green", 
                                "6" = "purple", "7" = "orange", "8" = "pink")) +
  scale_size_identity() +
  labs(title = "KPCA of Wine Quality Dataset (3 components)",
       x = "First Principal Component",
       y = "Second Principal Component",
       subtitle = "Point size represents the value on the third principal component") +
  theme_minimal() +
  guides(size = "none")