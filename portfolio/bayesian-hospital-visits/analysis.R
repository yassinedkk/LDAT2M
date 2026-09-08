require(R2WinBUGS)
require(coda)
require(rjags)
library(ggplot2)

#question 3
HospitalVisits <- read.delim("data/HospitalVisits.txt")
set.seed(123)
y <- HospitalVisits$y
age <- HospitalVisits$age
chronic <- HospitalVisits$chronic
hospital <- HospitalVisits$hospital

G <- length(unique(hospital))  
n <- length(y)
a0 <- 0.01
b0 <- 0.01
sigma2_beta <- 100
#initialisation
n_iter <- 25000
burn_in  <- 5000
beta <- matrix(0, nrow = n_iter, ncol = 3)
colnames(beta) <- c("beta0", "beta1", "beta2")
v <- matrix(1, nrow = n_iter, ncol = G)
alpha <- numeric(n_iter)
alpha[1] <- 1
prop_sd_beta <- c(0.05, 0.001, 0.05)
prop_sd_log_alpha <- 0.1

#MCMC univariate
for (t in 2:n_iter){
  beta_curr <- beta[t-1, ]
  v_curr <- v[t-1, ]
  alpha_curr <- alpha[t-1]
  eta <- beta_curr[1] + beta_curr[2]*age + beta_curr[3]*chronic
  
  # ---- Gibbs for v_g ----
  for(g in 1:G){
    idx <- which(hospital == g)
    s <- sum(y[idx]) + alpha_curr
    r <- sum(exp(eta[idx])) + alpha_curr
    v[t, g] <- rgamma(1, shape = s, rate = r)
  }
  
  v_g <- v[t, ]
  mu_curr <- v_g[hospital] * exp(eta)
  
  # ---- MH for beta ----
  for(j in 1:3){
    beta_prop <- beta_curr
    beta_prop[j] <- rnorm(1, beta_curr[j], prop_sd_beta[j])
    eta_prop <- beta_prop[1] + beta_prop[2]*age + beta_prop[3]*chronic
    mu_prop <- v_g[hospital] * exp(eta_prop)
    
    log_ratio <- sum(dpois(y, mu_prop, log = TRUE)) -
      sum(dpois(y, mu_curr, log = TRUE)) +
      dnorm(beta_prop[j], 0, sqrt(sigma2_beta), log = TRUE) -
      dnorm(beta_curr[j], 0, sqrt(sigma2_beta), log = TRUE)
    
    if (log(runif(1)) < log_ratio) {
      beta_curr <- beta_prop
      eta <- eta_prop
      mu_curr <- mu_prop
    }
    
  }
  beta[t, ] <- beta_curr
  # ---- MH for log(alpha) ----
  log_alpha_curr <- log(alpha_curr)
  log_alpha_prop <- rnorm(1, log_alpha_curr, prop_sd_log_alpha)
  alpha_prop <- exp(log_alpha_prop)
  
  log_post <- function(a, v_vals) {
    G * (a * log(a) - lgamma(a)) + (a - 1)*sum(log(v_vals)) - a * sum(v_vals) +
      dgamma(a, a0, b0, log = TRUE)
  }
  log_post_prop <- log_post(alpha_prop, v_g)
  log_post_curr <- log_post(alpha_curr, v_g)
  log_post_prop <- log_post_prop + log(alpha_prop)
  log_post_curr <- log_post_curr + log(alpha_curr)
  
  # Calcul du ratio d'acceptation
  prob <- min(1, exp(log_post_prop - log_post_curr))
  
  # Acceptation / rejet
  if (runif(1) <= prob) {
    alpha[t] <- alpha_prop
  } else {
    alpha[t] <- alpha_curr
  }
}

post_beta <- beta[(burn_in + 1):n_iter, ]
post_alpha <- alpha[(burn_in + 1):n_iter]

par(mfrow = c(2, 2))
plot(post_beta[,1], type = "l", main = "Traceplot β_0")
plot(post_beta[,2], type = "l", main = "Traceplot β_1")
plot(post_beta[,3], type = "l", main = "Traceplot β_2")
plot(post_alpha, type = "l", main = "Traceplot α")

# question 4
#a

library(HDInterval)
data_pros<-data.frame(post_beta,post_alpha)
col <- c("beta0", "beta1", "beta2", "post_alpha")

results <- t(apply(data_pros[, col], 2, function(x) {
  c(
    mean = mean(x),
    median = median(x),
    sd = sd(x),
    quantile(x, c(0.025,  0.975)),
    hdi = as.numeric(hdi(x, credMass = 0.95))
  )
}))
print(results)

par(mfrow = c(5, 6), mar = c(2, 2, 1, 1))  


for (g in 1:30) {
  hist(v[, g], 
       main = paste("v", g), 
       xlab = "", 
       col = "skyblue",
       breaks = 20,
       cex.main = 0.8)  
}

#b

n_sim <- 3000  
n <- length(y)
n_post <- nrow(post_beta)
y_rep <- matrix(NA, nrow = n_sim, ncol = n)

set.seed(123)
ids <- sample(1:n_post, size = n_sim)
print(ids)
#simuler y
for (s in 1:n_sim) {
  beta_s <- post_beta[ids[s], ]
  v_s <- v[ids[s], ]
  
  eta_s <- beta_s[1] + beta_s[2] * age + beta_s[3] * chronic
  mu_s <- v_s[hospital] * exp(eta_s)
  
  y_rep[s, ] <- rpois(n,mu_s)
}
#comparaison
df_obs <- as.data.frame(table(y) / length(y))
df_rep <- as.data.frame(table(as.vector(y_rep)) / length(as.vector(y_rep)))
names(df_obs) <- names(df_rep) <- c("value", "prob")
ggplot() +
  geom_col(data = df_obs, aes(x = as.numeric(value), y = prob),
           fill = "grey") +
  geom_line(data = df_rep, aes(x = as.numeric(value), y = prob),
            color = "blue", linewidth = 1) +
  labs(title = "Comparaison des probabilités observées vs simulées",
       x = "Nombre de visites", y = "Probabilité") +
  theme_minimal()

ggplot() +
  geom_histogram(aes(x = y, y = ..density..), 
                 bins = 30, fill = "grey", alpha = 0.6, position = "identity") +
  geom_histogram(aes(x = as.vector(y_rep), y = ..density..),
                 bins = 30, fill = "blue", alpha = 0.3, position = "identity") +
  labs(title = "Comparaison des distributions",
       x = "Nombre de visites", y = "Densité") +
  theme_minimal()

cat("Empirical mean:", mean(y), "\n")
cat("Predictive mean:", mean(y_rep), "\n")

cat("Empirical variance:", var(y), "\n")
cat("Predictive variance:", var(as.vector(y_rep)), "\n")

lower <- apply(y_rep, 2, quantile, probs = 0.025)
upper <- apply(y_rep, 2, quantile, probs = 0.975)
coverage <- mean(y >= lower & y <= upper)
cat("Couverture à 95% :", round(coverage, 3), "\n")

bias <- mean(colMeans(y_rep) - y)
cat("Biais  :", round(bias, 3), "\n")



# question 5 

library(rjags)

data_jags <- list(y = y, age = age, chronic = chronic, hospital = hospital, N = length(y), G = length(unique(hospital)))


para_inits <- function() {
  list(
    beta0 = rnorm(1),
    beta1 = rnorm(1),
    beta2 = rnorm(1),
    alpha = rgamma(1, 1, 1),
    v = rgamma(length(unique(hospital)), 1, 1)
  )
}

q5_jags <- "model {
  for (i in 1:N) {
    y[i] ~ dpois(mu[i])
    mu[i] <- v[hospital[i]] * exp(beta0 + beta1 * age[i] + beta2 * chronic[i])
  }
  
  beta0 ~ dnorm(0, 0.01)  
  beta1 ~ dnorm(0, 0.01)
  beta2 ~ dnorm(0, 0.01)
  
  for (g in 1:G) {
    v[g] ~ dgamma(alpha, alpha)
  }
  
 
  alpha ~ dgamma(0.01, 0.01)
}"

model_q5 <- jags.model(textConnection(q5_jags),
                    data = data_jags,
                    inits = para_inits,
                    n.chains = 1,
                    n.adapt = 1000)

samples_q5 <- coda.samples(model_q5, variable.names = c("beta0", "beta1", "beta2", "alpha"),n.iter = 20000, start = 5001)

# 5b
summary(samples_q5)
results_jags5 <- t(apply(as.matrix(samples_q5), 2, function(x) {
      c(mean = mean(x),
        median = median(x),
        sd = sd(x),
        quantile(x, c(0.025,  0.975)),
        hdi = as.numeric(hdi(x, credMass = 0.95))
        )
    }))
print(results_jags5)

comp2model <- round(rbind(results, results_jags5),4)
rownames(comp2model) <- c("beta0", "beta1", "beta2", "post_alpha", "post_alpha_jags",
                       "beta0_jags", "beta1_jags", "beta_2jags")

print(comp2model)





#6
q6_jags <- "model {

  for (i in 1:N) {
    y[i] ~ dpois(mu[i])
    mu[i] <- exp(u[hospital[i]]) * exp(beta0 + beta1 * age[i] + beta2 * chronic[i])
  }


  beta0 ~ dnorm(0, 0.01)
  beta1 ~ dnorm(0, 0.01)
  beta2 ~ dnorm(0, 0.01)


  for (g in 1:G) {
    u[g] ~ dnorm(0, tau)
  }


  tau ~ dgamma(0.01, 0.01)
}"

para_inits2 <- function() {
  list(
    beta0 = rnorm(1),
    beta1 = rnorm(1),
    beta2 = rnorm(1),
    tau = rgamma(1, 1, 1),
    u = rnorm(length(unique(hospital)), 0, 1)
  )
}

model_q6 <- jags.model(textConnection(q6_jags), 
                    data = data_jags, 
                    inits = para_inits2, 
                    n.chains = 1, 
                    n.adapt = 1000)

samples_q6 <- coda.samples(model_q6, variable.names = c("beta0", "beta1", "beta2", "tau"), n.iter = 20000, start = 5001)

results_jags6 <- t(apply(as.matrix(samples_q6), 2, function(x) {
  c(mean = mean(x),
    median = median(x),
    sd = sd(x),
    quantile(x, c(0.025,  0.975)),
    hdi = as.numeric(hdi(x, credMass = 0.95))
  )
}))
print(round(results_jags6,4))
