## code to prepare `DATASET` dataset goes here
library(evd)
library(mvtnorm)
library(tictoc)
library(tidyverse)

get_extremes <- function(data, num = n_save) {
  l_infinity <- apply(data, 1, max)
  indices    <- which(l_infinity %in% tail(sort(l_infinity), num))
  data[indices, ]
}

invert <- function(data) {
  1/data
}

set.seed(545365)
n <- 10000 # number of points in each dataset
nruns <- 10000 # number of datasets
p_save <- 0.05 # proportion of the dataset to save (i.e., 0.05 => top 5%)
n_save <- p_save * n

# initialize variables
dep_param <- runif(nruns) # dependence parameters
asym_param <- matrix(runif(2 * nruns), ncol = 2) # asym params

data_mixed_four <- array(NA, dim = c(4 * nruns, n_save, 2))
data_mixed_full <- list()
# order:logistic (1:nruns),
#       gaussian((nruns+1):(2nruns)),
#       invertedLog(..),
#       asymLog(...)

tic()
for (i in 1:nruns) {
  # logistic (AD)
  # 1-dep_param so that as a increases the dependence increases
  # c(1,1,1) is frechet, loc=0, scale=1, shape=1
  data_mixed_full[[i]] <-
    rbvevd(n, dep = (1 - dep_param[i]), model = "log", mar1 = c(1, 1, 1)) %>%
    pfrechet() %>%
    qexp()
  data_mixed_four[i, , ] <- get_extremes(data_mixed_full[[i]])

  # gaussian (AI) with correlation dep_param
  data_mixed_full[[(nruns + i)]] <-
    rmvnorm(n, sigma = matrix(c(1, dep_param[i], dep_param[i], 1), nrow = 2)) %>%
    pnorm() %>%
    qexp()
  data_mixed_four[nruns + i, , ] <- get_extremes(data_mixed_full[[nruns + i]])

  # invertedLog (AI)
  # inverting frechet makes it exponential
  data_mixed_full[[2 * nruns + i]] <-
    rmvevd(n, dep = (1 - dep_param[i]), model = "log", d = 2, mar = c(1, 1, 1)) %>%
    invert()
  data_mixed_four[2 * nruns + i, , ] <-
    get_extremes(data_mixed_full[[2 * nruns + i]])

  # asymmetric logistic (AD)
  data_mixed_full[[3 * nruns + i]] <-
    rbvevd(n, dep = 1 - dep_param[i],
           asy = asym_param[i, ],
           model = "alog",
           mar1 = c(1, 1, 1)) %>%
    pfrechet() %>%
    qexp()
  data_mixed_four[3 * nruns + i, , ] <-
    get_extremes(data_mixed_full[[3 * nruns + i]])

  if (i %% 1000 == 0) {
    print(paste("iteration: ", i))
    toc()
    tic()
  }
}


# plot a few of the resulting sets to see if they look reasonable
par(mfrow = c(1, 2))
plot(exp(data_mixed_four[996, , ])) # logistic
plot(exp(data_mixed_four[(1896 + nruns), , ])) # gaussian
plot(exp(data_mixed_four[(294 + 2 * nruns), , ])) # inverteLog
plot(exp(data_mixed_four[(837 + 3 * nruns), , ])) # asymLog



# make matrix to keep track of group, dependence value, chi, and chiBar
chi <- c(
  2 - 2^(1 - dep_param),
  rep(0, nruns),
  rep(0, nruns),
  (asym_param[, 1] + asym_param[, 2] -
     (asym_param[, 1]^(1 / dep_param) +
        asym_param[, 2]^(1 / dep_param))^dep_param)
)
chiBar <- c(
  rep(1, nruns),
  dep_param,
  2^dep_param - 1,
  rep(0, nruns)
)
indices_four <- data.frame(
  "original_index" = 1:(4 * nruns),
  "AD=0_AI=1" = c(rep(0, nruns), rep(1, nruns), rep(1, nruns), rep(0, nruns)),
  "dep_param" = rep(dep_param, 4),
  "chi" = chi,
  "chiBar" = chiBar,
  "asym_param1" = c(rep(0, 3*nruns), asym_param[,1]),
  "asym_param2" = c(rep(0, 3*nruns), asym_param[,2]),
  "model" = rep(c("logistic", "gaussian", "invertedLog", "asymLog"),
                each = nruns))
shuffle_four <- sample(1:(4 * nruns), 4 * nruns, rep = F)
indices_four[1:(4 * nruns), ] <- indices_four[shuffle_four, ]

# mix the datasets
data_mixed_four[1:(4 * nruns), , ] <- data_mixed_four[shuffle_four, , ]

# save data in mixed format with the indices
saveRDS(data_mixed_full,
        file = "./CSU/MSPHD/Research/AD_AI_NN_exploration/full_data_0630.rds")
saveRDS(data_mixed_four,
        file = "./CSU/MSPHD/Research/AD_AI_NN_exploration/full_data_subset_0630.rds")
saveRDS(indices_four,
        file = "./CSU/MSPHD/Research/AD_AI_NN_exploration/metadata_0630.rds")
saveRDS(shuffle_four,
        file = "./CSU/MSPHD/Research/AD_AI_NN_exploration/shuffle_data_0630.rds")

##################################
# ALL FOUR train-validate-test split
##################################

p_train <- 0.8
p_valid <- 0.1
p_test <- 1 - (p_train + p_valid)

# get indices to pull for train, etc.
train_ind_four <- sample(1:(4 * nruns),
                         p_train * 4 * nruns,
                         rep = F)
valid_ind_four <- sample(setdiff(1:(4 * nruns), train_ind_four),
                         p_valid * 4 * nruns,
                         rep = F)
test_ind_four <- setdiff(1:(4 * nruns), c(train_ind_four, valid_ind_four))

# get full datasets for test data so I can look at chi-plots when classified wrong
data_mixed_full_test <- list()
for (i in 1:4000) {
  data_mixed_full_test[[i]] <-
    data_mixed_full[[shuffle_four[test_ind_four[i]]]]
}
saveRDS(data_mixed_full_test,
        "../full_data_test_0630.rds")
rm(data_mixed_full)
rm(data_mixed_full_test)

# set up arrays and separate data
train_data_four <- array(NA, dim = c(4 * p_train * nruns, n_save, 2))
valid_data_four <- array(NA, dim = c(4 * p_valid * nruns, n_save, 2))
test_data_four <- array(NA, dim = c(4 * p_valid * nruns, n_save, 2))

train_data_four[1:(4 * p_train * nruns), , ] <- data_mixed_four[train_ind_four, , ]
valid_data_four[1:(4 * p_valid * nruns), , ] <- data_mixed_four[valid_ind_four, , ]
test_data_four[1:(4 * p_test * nruns), , ] <- data_mixed_four[test_ind_four, , ]

# get vector of labels
train_response_four <- indices_four[train_ind_four, 2]
valid_response_four <- indices_four[valid_ind_four, 2]
test_response_four <- indices_four[test_ind_four, 2]

train_indices_four <- indices_four[train_ind_four, ]
valid_indices_four <- indices_four[valid_ind_four, ]
test_indices_four <- indices_four[test_ind_four, ]

# save the dependence parameters used in the test datasets
# saved for NN scoring
test_dep_four <- test_indices_four[, "dep_param"]
chi_four <- test_indices_four[, "chi"]
chiBar_four <- test_indices_four[, "chiBar"]
chi_chiBar_four <- ifelse(test_indices_four[, "chi"] != 0,
                          test_indices_four[, "chi"],
                          test_indices_four[, "chiBar"])
asym_test <- test_indices_four[, c("asym_param1", "asym_param2")]

# save the rdata that we need
save(train_data_four, train_response_four,
     valid_data_four, valid_response_four,
     test_data_four, test_response_four,
     test_dep_four, chi_four, chiBar_four, chi_chiBar_four, asym_test,
     file = "../train_test_0630.Rdata"
)



usethis::use_data(DATASET, overwrite = TRUE)
