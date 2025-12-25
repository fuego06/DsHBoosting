## =========================================================
## FULL PIPELINE: SQUIRREL + MODEL + HIERARCHY + t-SNE PLOTS
## =========================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(igraph)
  library(keras)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(Rtsne)
})

## -----------------------------
## 0) Project sources & TF options
## -----------------------------
source("C:/Users/user/Desktop/Thesis Coding/data_processing.R")
auto_load_libraries()
source("C:/Users/user/Desktop/Thesis Coding/Hieararchical Algorithm.R")
source("C:/Users/user/Desktop/Thesis Coding/DeepHierarchy.R")
source("C:/Users/user/Desktop/Thesis Coding/MLMetric Evaluation.R")
source("C:/Users/user/Desktop/Thesis Coding/Model Layers.R")
source("C:/Users/user/Desktop/Thesis Coding/HyperParameter Tuning.R")
source("C:/Users/user/Desktop/Thesis Coding/Train_Flex_Boost.R")
source("C:/Users/user/Desktop/Thesis Coding/Helpers.R")


options(tensorflow.extract.one_based = FALSE)
options(tensorflow.extract.warn_negatives_pythonic = FALSE)

set.seed(1)

## -----------------------------
## 1) Load Squirrel + structural priors
## -----------------------------
base_dir <- "C:/Users/user/Desktop/Thesis Coding/wiki_text_min"

ds <- read_triplet_ds(base_dir, name = "squirrel")
X_raw <- as.matrix(ds$X) # N x F
y_raw <- ds$y # labels
y <- as.integer(factor(y_raw)) - 1L # 0..K-1
N <- nrow(X_raw)
Kc <- length(unique(y))

A_raw <- make_Araw(ds$edges, N) # sparse NxN adjacency

# Graph priors C, D
gf_all <- graph_feats_all(A_raw)
C_all <- as.numeric(gf_all$C[, 1]) # [0,1]
D_all <- as.numeric(gf_all$D[, 1]) # [0,1]

## -----------------------------
## 2) Train/val/test split + feature preprocessing
## -----------------------------
sp <- stratified_split_60_20_20(y, seed = 8L)
train_idx <- sp$train
val_idx <- sp$val
test_idx <- sp$test

# Fit TF???IDF + row L2 + PCA only on training nodes (no leakage)
prep <- fit_feature_prep(
  X_raw[train_idx, , drop = FALSE],
  use_tfidf = TRUE,
  l2_rows = TRUE,
  pca_dim = 256L
)

Xp_all <- apply_feature_prep(X_raw, prep) # N x d_pca

# Optional multi-hop augmentation (good for Squirrel)
A_feat <- row_norm(A_raw + Diagonal(N)) # add self-loops, row-normalize
Xaug_all <- augment_AX(
  Xp_all,
  A_feat,
  gamma2 = 0.25,
  post_l2 = TRUE
)

n_nodes <- N
n_features <- ncol(Xaug_all)

## -----------------------------
## 3) Build model (internal boosting + hierarchy)
## -----------------------------
alpha_vec <- rep(1.0 / Kc, Kc) # for focal loss class weights (example)

model <- build_e2e_boost_flexible(
  n_nodes = n_nodes,
  n_features = n_features,
  num_classes = Kc,
  T = 3L,
  init_shrink = c(0.90, 0.85, 0.80),
  k_top = 49L,
  tau_cos = 0.07,
  init_alpha_blend = 0.1,
  block_feature_dim = 112L,
  prev_emb_dim = 16L,
  gate_dim_target = 128L,
  block_depth = 6L,
  block_dropout = 0.28,
  lr = 3e-3,
  l2 = 1e-3,
  use_focal = TRUE,
  focal_gamma = 1.6,
  focal_alpha = alpha_vec,
  use_cluster_gate = TRUE,
  gate_beta = 12,
  gate_normalize_inputs = TRUE
)

## >>> IMPORTANT <<<
## At this point you should either TRAIN the model or LOAD trained weights:
## Example (if you saved weights):
## load_model_weights_hdf5(model, "C:/path/to/squirrel_weights.h5")

## -----------------------------
## 4) Build full Keras inputs for all nodes
## -----------------------------
build_full_inputs <- function(model, X_features, A, C_vec, D_vec) {
  n_nodes <- nrow(X_features)
  n_features <- ncol(X_features)
  
  X_in <- array(X_features, dim = c(1, n_nodes, n_features))
  A_in <- array(as.matrix(A), dim = c(1, n_nodes, n_nodes))
  C_in <- array(C_vec, dim = c(1, n_nodes, 1))
  D_in <- array(D_vec, dim = c(1, n_nodes, 1))
  
  if (length(model$inputs) == 4L) {
    # (X, A, C, D)
    list(X_in, A_in, C_in, D_in)
  } else if (length(model$inputs) == 5L) {
    # (X, A, C, D, M) ??? allow all edges for analysis
    M_in <- array(1, dim = c(1, n_nodes, n_nodes))
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

## -----------------------------
## 5) Helper: get hierarchy gate outputs g_i
## -----------------------------
get_hierarchy_gate_outputs <- function(model,
                                       inputs,
                                       layer_pattern = "hierarchy_gate|hierarch_gate") {
  layer_names <- sapply(model$layers, function(l) l$name)
  idx <- which(grepl(layer_pattern, layer_names, ignore.case = TRUE))[1]
  if (is.na(idx)) stop("Could not find HierarchyGate layer in model.")
  
  hg_layer <- model$layers[[idx]]
  hg_model <- keras_model(inputs = model$inputs, outputs = hg_layer$output)
  H_pred <- predict(hg_model, inputs) # (1, N, 1) or (N,1)
  
  if (length(dim(H_pred)) == 3L) {
    g_vec <- as.numeric(H_pred[1, , 1])
  } else if (length(dim(H_pred)) == 2L) {
    g_vec <- as.numeric(H_pred[, 1])
  } else {
    stop("Unexpected HierarchyGate tensor shape.")
  }
  g_vec
}

g_vec <- get_hierarchy_gate_outputs(model, inputs_all)

## -----------------------------
## 6) Helper: extract final embeddings (penultimate layer)
## -----------------------------
get_final_embeddings <- function(model, inputs) {
  layer_names <- sapply(model$layers, function(l) l$name)
  L <- length(model$layers)
  emb_layer_name <- model$layers[[L - 1L]]$name
  cat("Using penultimate layer as embedding layer:", emb_layer_name, "\n")
  
  emb_model <- keras_model(
    inputs = model$inputs,
    outputs = get_layer(model, emb_layer_name)$output
  )
  
  Z_pred <- predict(emb_model, inputs)
  if (length(dim(Z_pred)) == 3L) {
    Z_final <- Z_pred[1, , ]
  } else if (length(dim(Z_pred)) == 2L) {
    Z_final <- Z_pred
  } else {
    stop("Unexpected Z_pred shape:", paste(dim(Z_pred), collapse = " x "))
  }
  as.matrix(Z_final)
}

Z_final <- get_final_embeddings(model, inputs_all) # N x d

## -----------------------------
## 7) Probabilities and predictions from model
## -----------------------------
logits <- predict(model, inputs_all)

if (is.list(logits)) logits <- logits[[1]]

if (length(dim(logits)) == 3L) {
  logits_mat <- logits[1, , ] # N x K
} else if (length(dim(logits)) == 2L) {
  logits_mat <- logits # N x K
} else {
  stop("Unexpected logits shape: ", paste(dim(logits), collapse = " x "))
}

logits_mat <- as.matrix(logits_mat)
logits_shift <- logits_mat - apply(logits_mat, 1, max)
exp_logits <- exp(logits_shift)
P_final <- exp_logits / rowSums(exp_logits) # N x Kc

true_label_idx <- y + 1L
true_label <- factor(y)
pred_label_idx <- max.col(P_final)
pred_label <- factor(pred_label_idx - 1L)
correct <- (pred_label_idx - 1L) == y

## -----------------------------
## 8) Robust t-SNE on embeddings
## -----------------------------
Z_mat <- Z_final

good_rows <- apply(Z_mat, 1, function(r) all(is.finite(r)))
if (!all(good_rows)) {
  warning("Some nodes had NA/Inf in embeddings and were dropped for t-SNE.")
}
Z_tsne <- Z_mat[good_rows, , drop = FALSE]

N_tsne <- nrow(Z_tsne)
if (N_tsne < 10) {
  stop("Too few valid nodes for t-SNE after filtering: N_tsne = ", N_tsne)
}

perplexity_val <- min(30, max(5, floor((N_tsne - 1) / 3)))
cat("Running t-SNE on", N_tsne, "nodes with perplexity =", perplexity_val, "\n")

set.seed(123)
ts <- Rtsne(
  Z_tsne,
  perplexity = perplexity_val,
  check_duplicates = FALSE,
  verbose = TRUE
)

idx_kept <- which(good_rows)

df_tsne <- data.frame(
  x = ts$Y[, 1],
  y = ts$Y[, 2],
  true_label = factor(y[idx_kept]),
  pred_label = factor(pred_label[idx_kept]),
  correct = factor(correct[idx_kept],
                   levels = c(FALSE, TRUE),
                   labels = c("wrong", "correct")),
  depth = as.numeric(D_all[idx_kept]),
  centrality = as.numeric(C_all[idx_kept]),
  gate = as.numeric(g_vec[idx_kept])
)

## -----------------------------
## 9) t-SNE plots (hierarchy + prediction)
## -----------------------------

# (a) TRUE labels
p_tsne_true <- ggplot(df_tsne, aes(x = x, y = y, colour = true_label)) +
  geom_point(alpha = 0.85, size = 1.2) +
  labs(x = "t-SNE 1", y = "t-SNE 2", colour = "True class") +
  ggtitle("t-SNE of final embeddings (true labels)") +
  theme_minimal(base_size = 11)

# (b) PREDICTED labels, correct vs wrong
p_tsne_pred <- ggplot(df_tsne,
                      aes(x = x, y = y,
                          colour = pred_label,
                          shape = correct)) +
  geom_point(alpha = 0.9, size = 1.4) +
  labs(x = "t-SNE 1", y = "t-SNE 2",
       colour = "Predicted class",
       shape = "Prediction") +
  ggtitle("t-SNE of final embeddings (predicted class, correct vs wrong)") +
  theme_minimal(base_size = 11)

# (c) Depth prior
p_tsne_depth <- ggplot(df_tsne, aes(x = x, y = y, colour = depth)) +
  geom_point(alpha = 0.9, size = 1.3) +
  scale_colour_viridis_c(option = "plasma") +
  labs(x = "t-SNE 1", y = "t-SNE 2", colour = "Depth D_i") +
  ggtitle("t-SNE of final embeddings (depth prior D_i)") +
  theme_minimal(base_size = 11)

# (d) Hierarchy gate
p_tsne_gate <- ggplot(df_tsne, aes(x = x, y = y, colour = gate)) +
  geom_point(alpha = 0.9, size = 1.3) +
  scale_colour_viridis_c(option = "magma") +
  labs(x = "t-SNE 1", y = "t-SNE 2", colour = "Gate g_i") +
  ggtitle("t-SNE of final embeddings (hierarchy gate g_i)") +
  theme_minimal(base_size = 11)

print(p_tsne_true)
print(p_tsne_pred)
print(p_tsne_depth)
print(p_tsne_gate)

## -----------------------------
## 10) Simple hierarchy scatter plots
## -----------------------------
deg_vec <- as.numeric(Matrix::rowSums(A_raw))

df_hier <- data.frame(
  node = seq_len(N),
  depth = D_all,
  centrality = C_all,
  degree = deg_vec,
  label = factor(y),
  gate = g_vec
)

# Depth vs gate
p_depth_gate <- ggplot(df_hier, aes(x = depth, y = gate, colour = label)) +
  geom_point(alpha = 0.6, size = 0.9) +
  labs(x = "Depth prior D_i", y = "Hierarchy gate g_i", colour = "Class") +
  ggtitle("Depth vs hierarchy gate on Squirrel") +
  theme_minimal(base_size = 11)

# Centrality vs gate
p_central_gate <- ggplot(df_hier, aes(x = centrality, y = gate, colour = label)) +
  geom_point(alpha = 0.6, size = 0.9) +
  labs(x = "Centrality prior C_i", y = "Hierarchy gate g_i", colour = "Class") +
  ggtitle("Centrality vs hierarchy gate on Squirrel") +
  theme_minimal(base_size = 11)

# Depth vs centrality, coloured by gate
p_depth_central_gate <- ggplot(df_hier, aes(x = depth, y = centrality, colour = gate)) +
  geom_point(alpha = 0.8, size = 0.9) +
  scale_colour_viridis_c(option = "plasma") +
  labs(x = "Depth prior D_i", y = "Centrality prior C_i", colour = "Gate g_i") +
  ggtitle("Structural hierarchy and learned gate") +
  theme_minimal(base_size = 11)

print(p_depth_gate)
print(p_central_gate)
print(p_depth_central_gate)