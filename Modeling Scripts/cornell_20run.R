# =========================
# 20-run Cora experiment (E2E-Boost, your params)
# =========================

suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
})

# -------- CONFIG --------
dataset_name <- "Cornell"
# Change this if your data folder is elsewhere:
base_dir <- "C:/Users/user/Desktop/Thesis Coding/wekb_text_min" # contains edges.txt, features.csv, labels.csv
out_root <- file.path("runs", paste0(dataset_name, "_e2eBoost"))
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

seeds <- 1:20

# -------- HELPERS --------
# Robust dataset loader: supports either read_three() or read_triplet_ds()
load_dataset <- function(base_dir, dataset_name) {
  if (exists("read_three", mode = "function")) {
    ds <- read_three(base_dir, dataset_name)
  } else if (exists("read_triplet_ds", mode = "function")) {
    ds <- read_triplet_ds(base_dir, dataset_name)
    # harmonize y to 0-based ints
    if (!is.integer(ds$y)) ds$y <- as.integer(factor(ds$y)) - 1L
  } else {
    stop("Neither read_three() nor read_triplet_ds() was found. Please source your data utils.")
  }
  ds
}

# Robust stratified split wrapper
do_stratified_split <- function(y, seed) {
  if (exists("stratified_60_20_20", mode = "function")) {
    stratified_60_20_20(y, seed = as.integer(seed))
  } else if (exists("stratified_split_60_20_20", mode = "function")) {
    stratified_split_60_20_20(y, seed = as.integer(seed))
  } else {
    stop("No stratified split function found (stratified_60_20_20 or stratified_split_60_20_20).")
  }
}

save_vec <- function(path, v) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.table(matrix(v, ncol = 1), file = path, row.names = FALSE, col.names = FALSE)
}

macro_f1_from_labels <- function(y_true, y_pred) {
  cls <- sort(unique(y_true))
  cm <- table(factor(y_true, levels = cls), factor(y_pred, levels = cls))
  prec <- diag(cm) / pmax(colSums(cm), 1)
  rec <- diag(cm) / pmax(rowSums(cm), 1)
  f1 <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  mean(f1)
}

# -------- LOAD ONCE (graph, raw features) --------
ds <- load_dataset(base_dir, dataset_name)
X <- as.matrix(ds$X)
y <- ds$y
if (!is.integer(y)) y <- as.integer(factor(y)) - 1L # ensure 0-based labels

N <- nrow(X)
Kc <- length(unique(y))

if (!exists("make_Araw", mode = "function")) stop("make_Araw() not found. Please source your graph utils.")
A_raw <- make_Araw(ds$edges, N) # undirected, binary, no self-loops

if (!exists("graph_feats_all", mode = "function")) stop("graph_feats_all() not found. Please source your graph utils.")
gf <- graph_feats_all(+A_raw); C_all <- gf$C; D_all <- gf$D

# Full-graph allowance mask (if your model uses it)
M_all <- matrix(1, N, N); diag(M_all) <- 0

# -------- Write hyperparameters (once) --------
hparams_path <- file.path(out_root, paste0(dataset_name, "_hparams.txt"))
if (!file.exists(hparams_path)) {
  cat(
    sprintf(
      "dataset=%s
T = 3L,
 init_shrink = c(0.90, 0.85, 0.80),
 k_top = 10L, tau_cos = 0.20, init_alpha_blend = 0.9,
 block_feature_dim = 64L, prev_emb_dim = 16L, gate_dim_target = 80L,
 block_depth = 6L, block_dropout = 0.20,
 lr = 3e-3, l2 = 1e-3,
 use_focal = TRUE, focal_gamma = 1.6, focal_alpha = alpha_vec,
 use_cluster_gate = TRUE, gate_beta = 10, gate_normalize_inputs = TRUE
  rounds = 3L, epochs_per_round = 100L,
 hard_thresh = 0.45, up_factor = 2.5,
 verbose = 1L, es_patience = 35L,
 rlrop = TRUE, rlrop_factor = 0.5, rlrop_patience = 6L, rlrop_min_lr = 1e-5
", dataset_name),
    file = hparams_path
  )
}

# -------- Results CSV header --------
results_csv <- file.path(out_root, paste0(dataset_name, "_20runs_results.csv"))
write.table(
  data.frame(seed=integer(), val_acc=double(), test_acc=double(),
             val_macro_f1=double(), test_macro_f1=double()),
  file = results_csv, sep = ",", row.names = FALSE, col.names = TRUE
)

# -------- MAIN 20-RUN LOOP --------
all_rows <- list()

for (seed in seeds) {
  message(sprintf("\n===== %s | seed %d =====", dataset_name, seed))
  
  # Clean TF/Keras state & set seeds
  keras::k_clear_session(); gc()
  set.seed(seed)
  try({ tensorflow::tf$random$set_seed(as.integer(seed)) }, silent = TRUE)
  
  # ----- stratified 60/20/20 split -----
  sp <- do_stratified_split(y, seed)
  train_idx <- sp$train; val_idx <- sp$val; test_idx <- sp$test
  
  # Save split files (1-based indices, as in R)
  split_dir <- file.path(out_root, sprintf("seed_%02d", seed))
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)
  save_vec(file.path(split_dir, "train_idx.txt"), train_idx)
  save_vec(file.path(split_dir, "val_idx.txt"), val_idx)
  save_vec(file.path(split_dir, "test_idx.txt"), test_idx)
  
  # ----- feature prep: fit on TRAIN only -----
  pca_dim <- min(128L, length(train_idx) - 1L)
  if (!exists("fit_feature_prep", mode = "function") ||
      !exists("apply_feature_prep", mode = "function")) {
    stop("fit_feature_prep()/apply_feature_prep() not found. Please source your feature utils.")
  }
  
  prep <- fit_feature_prep(
    X[train_idx, , drop = FALSE],
    use_tfidf = TRUE, l2_rows = TRUE, pca_dim = pca_dim
  )
  Xp_all <- apply_feature_prep(X, prep) # (N, d_pca)
  
  # Optional AX augmentation (kept for future use; inputs use Xp_all)
  if (!exists("row_norm", mode = "function")) stop("row_norm() not found in your utils.")
  A_feat <- row_norm(+A_raw); diag(A_feat) <- 1
  X_ax <- A_feat %*% Xp_all
  if (!exists("row_l2", mode = "function")) stop("row_l2() not found in your utils.")
  Xaug_all <- row_l2(cbind(Xp_all, 0.12 * X_ax))
  
  # ----- build inputs (batch=1) -----
  inputs_all <- list(
    X_in = array(Xp_all, dim = c(1, N, ncol(Xp_all))),
    A_in = array(as.matrix(A_raw), dim = c(1, N, N)),
    C_in = array(C_all, dim = c(1, N, 1)),
    D_in = array(D_all, dim = c(1, N, 1)),
    M_in = array(M_all, dim = c(1, N, N))
  )
  
  # ----- labels & masks -----
  y_batch <- array(keras::to_categorical(y, Kc), dim = c(1, N, Kc))
  sw_train <- array(as.numeric(seq_len(N) %in% train_idx), dim = c(1, N, 1))
  sw_val <- array(as.numeric(seq_len(N) %in% val_idx), dim = c(1, N, 1))
  
  # focal alpha from TRAIN distribution (per-seed)
  alpha_counts <- tabulate(y[train_idx] + 1L, nbins = Kc)
  inv_freq <- 1 / pmax(alpha_counts, 1)
  alpha_vec <- as.numeric(inv_freq / mean(inv_freq))
  
  # ----- build model (YOUR PARAMS) -----
  keras::k_clear_session()
  if (!exists("build_e2e_boost_flexible", mode = "function"))
    stop("build_e2e_boost_flexible() not found. Please source your model code.")
  if (!exists("train_e2e_boost", mode = "function"))
    stop("train_e2e_boost() not found. Please source your training code.")
  
  model_full <- build_e2e_boost_flexible(
    n_nodes = N, n_features = ncol(Xp_all), num_classes = Kc,
    T = 3L,
    init_shrink = c(0.90, 0.85, 0.80),
    k_top = 10L, tau_cos = 0.20, init_alpha_blend = 0.2,
    block_feature_dim = 64L, prev_emb_dim = 16L, gate_dim_target = 80L,
    block_depth = 6L, block_dropout = 0.20,
    lr = 3e-3, l2 = 1e-3,
    use_focal = TRUE, focal_gamma = 1.6, focal_alpha = alpha_vec,
    use_cluster_gate = TRUE, gate_beta = 10, gate_normalize_inputs = TRUE
  )
  
  model_full <- train_e2e_boost(
    model_full,
    inputs_trainval = inputs_all, 
    y_batch = y_batch,
    train_mask = sw_train, val_mask = sw_val,
    rounds = 3L, epochs_per_round = 100L,
    hard_thresh = 0.45, up_factor = 2.5,
    verbose = 1L, es_patience = 35L,
    rlrop = TRUE, rlrop_factor = 0.5, rlrop_patience = 6L, rlrop_min_lr = 1e-5
   
  )
  
  # ----- evaluate -----
  p_list <- predict(model_full, inputs_all, verbose = 0)
  P <- if (is.list(p_list)) p_list[[1]] else p_list
  if (length(dim(P)) == 2L) P <- array(P, dim = c(1, nrow(P), ncol(P)))
  P <- P[1, , ] # N x K
  
  pred_val <- max.col(P[val_idx, , drop = FALSE]) - 1L
  pred_test <- max.col(P[test_idx, , drop = FALSE]) - 1L
  
  acc_val <- mean(pred_val == y[val_idx])
  acc_test <- mean(pred_test == y[test_idx])
  
  f1_val <- macro_f1_from_labels(y[val_idx], pred_val)
  f1_test <- macro_f1_from_labels(y[test_idx], pred_test)
  
  message(sprintf("seed %02d | VAL acc=%.4f F1=%.4f | TEST acc=%.4f F1=%.4f",
                  seed, acc_val, f1_val, acc_test, f1_test))
  
  # save per-seed predictions and metrics
  write.table(
    data.frame(node = test_idx, y_true = y[test_idx], y_pred = pred_test),
    file = file.path(split_dir, "pred_test.csv"),
    sep = ",", row.names = FALSE, col.names = TRUE
  )
  write.table(
    data.frame(node = val_idx, y_true = y[val_idx], y_pred = pred_val),
    file = file.path(split_dir, "pred_val.csv"),
    sep = ",", row.names = FALSE, col.names = TRUE
  )
  
  # append to results CSV
  one <- data.frame(seed = seed,
                    val_acc = acc_val, test_acc = acc_test,
                    val_macro_f1 = f1_val, test_macro_f1 = f1_test)
  write.table(one, file = results_csv, sep = ",",
              row.names = FALSE, col.names = FALSE, append = TRUE)
  all_rows[[length(all_rows)+1]] <- one
}

# -------- SUMMARY --------
res <- do.call(rbind, all_rows)
mean_sd <- function(v) c(mean = mean(v), sd = sd(v))
summary_tbl <- rbind(
  val_acc = mean_sd(res$val_acc),
  test_acc = mean_sd(res$test_acc),
  val_macro_f1 = mean_sd(res$val_macro_f1),
  test_macro_f1 = mean_sd(res$test_macro_f1)
)
print(round(summary_tbl, 4))

write.table(
  data.frame(metric = rownames(summary_tbl),
             mean = summary_tbl[, "mean"],
             sd = summary_tbl[, "sd"]),
  file = file.path(out_root, "summary_mean_sd.csv"),
  sep = ",", row.names = FALSE, col.names = TRUE
)

cat("\nAll done. Results:\n",
    "- per-seed splits & preds: ", out_root, "/seed_XX/\n",
    "- hyperparameters: ", hparams_path, "\n",
    "- per-run metrics CSV: ", results_csv, "\n",
    "- summary mean??sd CSV: ", file.path(out_root, 'summary_mean_sd.csv'), "\n", sep = "")