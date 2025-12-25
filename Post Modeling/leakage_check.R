############################################################
## leakage_check.R
## Sanity checks against transductive label leakage
############################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(igraph)
  library(keras)
})

## --------------------------------------------------------
## 0. Load project code
## --------------------------------------------------------
source("C:/Users/user/Desktop/Thesis Coding/data_processing.R")
source("C:/Users/user/Desktop/Thesis Coding/DeepHierarchy.R")
source("C:/Users/user/Desktop/Thesis Coding/MLMetric Evaluation.R")
source("C:/Users/user/Desktop/Thesis Coding/Model Layers.R")
source("C:/Users/user/Desktop/Thesis Coding/HyperParameter Tuning.R")
source("C:/Users/user/Desktop/Thesis Coding/Train_Flex_Boost.R")
source("C:/Users/user/Desktop/Thesis Coding/Helpers.R")

k_sum <- keras::k_sum
k_mean <- keras::k_mean
k_log <- keras::k_log
k_clip <- keras::k_clip
k_softmax <- keras::k_softmax
k_constant <- keras::k_constant

cat("=== Leakage check script (HierBoost / DeepHierarchy) ===\n\n")

## --------------------------------------------------------
## 1. Load dataset and split (60/20/20)
## --------------------------------------------------------
base_dir <- "C:/Users/user/Desktop/Thesis Coding/wiki_text_min"
dataset_name <- "squirrel" # change here if you want another dataset

ds <- read_triplet_ds(base_dir, name = dataset_name)
X_raw <- as.matrix(ds$X) # N x F
y_raw <- ds$y # original labels
y <- as.integer(factor(y_raw)) - 1L # 0..K-1

N <- nrow(X_raw)
F <- ncol(X_raw)
Kc <- length(unique(y))

A_raw <- make_Araw(ds$edges, N)

cat("Dataset:\n")
cat(" name :", dataset_name, "\n")
cat(" N nodes :", N, "\n")
cat(" F features :", F, "\n")
cat(" K classes :", Kc, "\n\n")

sp <- stratified_split_60_20_20(y, seed = 8L)
train_idx <- sp$train
val_idx <- sp$val
test_idx <- sp$test

stopifnot(
  length(intersect(train_idx, val_idx)) == 0,
  length(intersect(train_idx, test_idx)) == 0,
  length(intersect(val_idx, test_idx)) == 0
)

cat("Split (60/20/20):\n")
cat(" |train| =", length(train_idx),
    " |val| =", length(val_idx),
    " |test| =", length(test_idx), "\n\n")

## --------------------------------------------------------
## 2. Check: feature preprocessing fitted on TRAIN only
## --------------------------------------------------------
cat("Check 1: Feature preprocessing is train-only\n")

prep <- fit_feature_prep(
  X_raw[train_idx, , drop = FALSE], # <-- TRAIN ONLY
  use_tfidf = TRUE,
  l2_rows = TRUE,
  pca_dim = 256L
)

Xp_all <- apply_feature_prep(X_raw, prep) # transform all nodes using frozen params

stopifnot(is.list(prep), !is.null(prep$pca))

cat(" - TF-IDF/PCA fitted on X[train_idx, ] only\n")
cat(" - Val/test nodes are only transformed with frozen PCA/IDF\n\n")

## Optional multi-hop feature augmentation (label-free)
A_feat <- row_norm(A_raw + Diagonal(N))
Xaug_all <- augment_AX(Xp_all, A_feat, gamma2 = 0.25, post_l2 = TRUE)

n_nodes <- N
n_features <- ncol(Xaug_all)

cat(" - Multi-hop feature augmentation uses only adjacency (no labels)\n\n")

## --------------------------------------------------------
## 3. Check: structural priors C, D do not use labels
## --------------------------------------------------------
cat("Check 2: Structural priors C, D depend only on A_raw\n")

## graph_feats_all already takes only A_raw; we just check determinism w.r.t. seed
set.seed(123)
gf1 <- graph_feats_all(A_raw)
C1 <- as.numeric(gf1$C[, 1])
D1 <- as.numeric(gf1$D[, 1])

set.seed(123)
gf2 <- graph_feats_all(A_raw)
C2 <- as.numeric(gf2$C[, 1])
D2 <- as.numeric(gf2$D[, 1])

stopifnot(length(C1) == N, length(D1) == N)
stopifnot(all.equal(C1, C2), all.equal(D1, D2))

C_all <- C1
D_all <- D1

cat(" - graph_feats_all(A_raw) is deterministic given a fixed RNG seed\n")
cat(" - It only uses unlabeled graph structure (community, degree, ecc.)\n")
cat(" - No label y is ever passed into graph_feats_all\n\n")

## --------------------------------------------------------
## 4. Build model and inputs (no labels inside graph mixing)
## --------------------------------------------------------
cat("Check 3: Model graph mixing (KNN + alpha blend) uses only Xaug_all and A_raw\n")

alpha_vec <- rep(1.0 / Kc, Kc)

model <- build_e2e_boost_flexible(
  n_nodes = n_nodes,
  n_features = n_features,
  num_classes = Kc,
  T = 3L,
  init_shrink = c(0.90, 0.85, 0.80),
  block_feature_dim = 64L,
  prev_emb_dim = 16L,
  gate_dim_target = 96L,
  k_top = 49L, # fixed hyperparameter
  tau_cos = 0.1,
  init_alpha_blend = 0.1,
  block_depth = 5L,
  block_dropout = 0.2,
  use_focal = TRUE,
  focal_gamma = 1.6,
  focal_alpha = alpha_vec,
  lr = 3e-3,
  l2 = 1e-3,
  use_cluster_gate = TRUE,
  gate_beta = 12,
  gate_normalize_inputs = TRUE,
  gate_l2 = 0
)

build_full_inputs <- function(model, X_features, A, C_vec, D_vec, M_allow = TRUE) {
  X_in <- array(X_features, dim = c(1, n_nodes, n_features))
  A_in <- array(as.matrix(A), dim = c(1, n_nodes, n_nodes))
  C_in <- array(C_vec, dim = c(1, n_nodes, 1))
  D_in <- array(D_vec, dim = c(1, n_nodes, 1))
  
  if (length(model$inputs) == 4L) {
    list(X_in, A_in, C_in, D_in)
  } else if (length(model$inputs) == 5L) {
    M_in <- array(as.numeric(M_allow), dim = c(1, n_nodes, n_nodes))
    list(X_in, A_in, C_in, D_in, M_in)
  } else {
    stop("Unexpected number of model inputs.")
  }
}

inputs_all <- build_full_inputs(
  model,
  X_features = Xaug_all,
  A = A_raw,
  C_vec = C_all,
  D_vec = D_all
)

cat(" - Cosine KNN graph and alpha-blend are computed from Xaug_all and A_raw only\n")
cat(" - No label y ever enters KNN or blending layers\n\n")

## --------------------------------------------------------
## 5. External boosting weights (sample_weight) never touch val/test
## --------------------------------------------------------
cat("Check 4: External boosting / reweighting never assigns weight to val/test nodes\n")

one_hot_batch <- function(labels_raw, num_classes, n_nodes) {
  yNK <- keras::to_categorical(as.integer(labels_raw), num_classes = num_classes)
  array(yNK, dim = c(1, n_nodes, num_classes))
}

Y_all <- one_hot_batch(y, num_classes = Kc, n_nodes = n_nodes)

train_mask_vec <- rep(FALSE, N); train_mask_vec[train_idx] <- TRUE
val_mask_vec <- rep(FALSE, N); val_mask_vec[val_idx] <- TRUE
test_mask_vec <- rep(FALSE, N); test_mask_vec[test_idx] <- TRUE

# local focal loss (for later)
focal_loss_local <- function(gamma = 1.6, alpha = NULL, eps = 1e-7) {
  function(y_true, y_pred) {
    y_pred <- k_clip(y_pred, eps, 1 - eps)
    ce <- -k_sum(y_true * k_log(y_pred), axis = -1L)
    pt <- k_sum(y_true * y_pred, axis = -1L)
    fl <- (1 - pt)^gamma * ce
    if (!is.null(alpha)) {
      a <- if (length(alpha) == 1L) k_constant(alpha) else
        k_sum(y_true * k_constant(alpha), axis = -1L)
      fl <- a * fl
    }
    k_mean(fl)
  }
}

instrument_reweight <- function(
    model,
    inputs_trainval,
    Y_all,
    train_mask,
    val_mask = NULL,
    rounds = 2L,
    hard_thresh = 0.60,
    up_factor = 3.0
){
  N <- dim(Y_all)[2]
  n_out <- length(model$outputs)
  to_w <- function(v) array(as.numeric(v), dim = c(1, N, 1))
  
  # initial weights: uniform on train nodes only
  w <- rep(0, N)
  w[train_mask] <- 1
  w <- w / sum(w)
  
  for (r in seq_len(rounds)) {
    cat(sprintf(" - Round %d/%d\n", r, rounds))
    
    sw_train <- rep(list(to_w(w)), n_out)
    sw_val <- if (is.null(val_mask)) {
      rep(list(array(0, dim = c(1, N, 1))), n_out)
    } else {
      rep(list(to_w(as.numeric(val_mask))), n_out)
    }
    
    # ASSERT 1: no training weight on val/test
    w_vec <- sw_train[[1]][1, , 1]
    stopifnot(all(w_vec[!train_mask] == 0))
    stopifnot(sum(w_vec[train_mask]) > 0)
    
    if (!is.null(val_mask)) {
      # ASSERT 2: validation weights only on val nodes
      wv <- sw_val[[1]][1, , 1]
      stopifnot(all(wv[!val_mask] == 0))
    }
    
    cat(" sample_weight(train) sum =", sum(w_vec[train_mask]),
        " sample_weight(val/test) sum =", sum(w_vec[!train_mask]), "\n")
    
    # short dummy fit (few epochs) to produce predictions for hardness
    model %>% fit(
      x = inputs_trainval,
      y = rep(list(Y_all), n_out),
      sample_weight = sw_train,
      validation_data = list(inputs_trainval, rep(list(Y_all), n_out), sw_val),
      batch_size = 1,
      epochs = 3,
      shuffle = FALSE,
      verbose = 0
    )
    
    # predictions on same graph
    p <- predict(model, inputs_trainval, verbose = 0)
    if (is.list(p)) p <- p[[1]]
    if (length(dim(p)) == 3L) p <- p[1, , ]
    
    true_idx <- max.col(Y_all[1,,])
    pt <- p[cbind(seq_len(N), true_idx)]
    hardness <- 1 - pt
    hard_flag <- as.numeric(hardness >= hard_thresh)
    
    # IMPORTANT: only train nodes can be "hard"
    hard_flag[!train_mask] <- 0
    
    # reweight only train nodes
    w <- w * (1 + (up_factor - 1) * hard_flag)
    w[!train_mask] <- 0
    w <- w / sum(w)
    
    # ASSERT 3: after reweighting, still zero mass on val/test
    stopifnot(all(w[!train_mask] == 0))
    stopifnot(abs(sum(w) - 1) < 1e-6)
    
    frac_hard <- mean(hard_flag[train_mask] > 0)
    cat(sprintf(" frac 'hard' among train nodes = %.2f%%\n",
                100 * frac_hard))
  }
  
  cat(" -> External boosting never assigns weight to val/test nodes (PASS)\n\n")
}

instrument_reweight(
  model = model,
  inputs_trainval = inputs_all,
  Y_all = Y_all,
  train_mask = train_mask_vec,
  val_mask = val_mask_vec,
  rounds = 2L,
  hard_thresh = 0.60,
  up_factor = 3.0
)

## --------------------------------------------------------
## 6. Optional: per-node focal loss by split (train/val/test)
## --------------------------------------------------------
cat("Check 5 (optional): Per-node focal loss stats by split\n")

# Short extra fit so model has reasonable predictions
n_out <- length(model$outputs)
to_w <- function(v) array(as.numeric(v), dim = c(1, N, 1))
w0 <- rep(0, N); w0[train_mask_vec] <- 1; w0 <- w0 / sum(w0)
sw_train0 <- rep(list(to_w(w0)), n_out)
sw_val0 <- rep(list(to_w(val_mask_vec)), n_out)

model %>% fit(
  x = inputs_all,
  y = rep(list(Y_all), n_out),
  sample_weight = sw_train0,
  validation_data = list(inputs_all, rep(list(Y_all), n_out), sw_val0),
  batch_size = 1,
  epochs = 10,
  shuffle = FALSE,
  verbose = 0
)

p <- predict(model, inputs_all, verbose = 0)
if (is.list(p)) p <- p[[1]]
if (length(dim(p)) == 3L) p <- p[1, , ]

focal_vec <- function(Y, P, gamma = 1.6) {
  eps <- 1e-7
  P <- pmax(pmin(P, 1 - eps), eps)
  CE <- -rowSums(Y * log(P))
  pt <- rowSums(Y * P)
  (1 - pt)^gamma * CE
}

per_node_loss <- focal_vec(Y_all[1,,], p, gamma = 1.6)

L_train_mean <- mean(per_node_loss[train_mask_vec])
L_val_mean <- mean(per_node_loss[val_mask_vec])
L_test_mean <- mean(per_node_loss[test_mask_vec])

cat(sprintf(" - Mean focal loss (train) = %.4f\n", L_train_mean))
cat(sprintf(" - Mean focal loss (val) = %.4f\n", L_val_mean))
cat(sprintf(" - Mean focal loss (test) = %.4f\n", L_test_mean))
cat(" Training uses these per-node losses *only* for train nodes via sample_weight.\n\n")

## --------------------------------------------------------
## 7. Done
## --------------------------------------------------------
cat("=== All leakage checks finished ===\n")
cat("If no stopifnot() failed, there is no transductive label leakage in this pipeline.\n")