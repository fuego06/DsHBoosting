suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
})

# ------------------- CONFIG -------------------
dataset_name <- "Actor"
base_dir <- "./actor_text_min" # folder with edges.txt, features.csv, labels.csv
out_root <- file.path("runs", paste0(dataset_name, "_e2eBoost"))
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

seeds <- 1:20

# ------------------- LOAD ONCE -------------------
ds <- read_three(base_dir, dataset_name)
X <- as.matrix(ds$X); y <- ds$y
N <- nrow(X); Kc <- length(unique(y))

A_raw <- make_Araw(ds$edges, N) # undirected, binary, no self-loops
gf <- graph_feats_all(+A_raw); C_all <- gf$C; D_all <- gf$D

# static mask for KNN allowance (we keep full-graph allowed; your model ignores M_in)
M_all <- matrix(1, N, N); diag(M_all) <- 0

# -------------- HELPERS --------------
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

# Write hyperparameters (once)
hparams_path <- file.path(out_root, paste0(dataset_name, "_hparams.txt"))
if (!file.exists(hparams_path)) {
  cat(
    sprintf(
      "dataset=%s
T=3
init_shrink=[0.90,0.85,0.80]
k_top=120
tau_cos=0.2
init_alpha_blend=0.05
block_feature_dim=64
prev_emb_dim=16
gate_dim_target=96
block_depth=5
block_dropout=0.2
lr=5e-3
l2=1e-4
use_focal=TRUE
focal_gamma=2
use_cluster_gate=TRUE
gate_beta=10
gate_normalize_inputs=TRUE
rounds=5
epochs_per_round=200
hard_thresh=0.45
up_factor=4.0
es_patience=40
rlrop=TRUE
rlrop_factor=0.5
rlrop_patience=7
rlrop_min_lr=1e-5
", dataset_name),
    file = hparams_path
  )
}

# Results CSV header
results_csv <- file.path(out_root, paste0(dataset_name, "_20runs_results.csv"))
write.table(
  data.frame(seed=integer(), val_acc=double(), test_acc=double(),
             val_macro_f1=double(), test_macro_f1=double()),
  file = results_csv, sep = ",", row.names = FALSE, col.names = TRUE
)

# -------------- MAIN LOOP --------------
all_rows <- list()

for (seed in seeds) {
  message(sprintf("\n===== %s | seed %d =====", dataset_name, seed))
  
  # clean TF/Keras state
  keras::k_clear_session(); gc()
  set.seed(seed)
  try({ tensorflow::tf$random$set_seed(as.integer(seed)) }, silent = TRUE)
  
  # ----- split (stratified 60/20/20) -----
  sp <- stratified_60_20_20(y, seed = as.integer(seed))
  train_idx <- sp$train; val_idx <- sp$val; test_idx <- sp$test
  
  # Save split files (1-based indices, as in R)
  split_dir <- file.path(out_root, sprintf("seed_%02d", seed))
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)
  save_vec(file.path(split_dir, "train_idx.txt"), train_idx)
  save_vec(file.path(split_dir, "val_idx.txt"), val_idx)
  save_vec(file.path(split_dir, "test_idx.txt"), test_idx)
  
  # ----- feature prep per-train -----
  pca_dim <- min(256L, length(train_idx) - 1L)
  prep <- fit_feature_prep(
    X[train_idx, , drop = FALSE],
    use_tfidf = TRUE, l2_rows = TRUE, pca_dim = pca_dim
  )
  Xp_all <- apply_feature_prep(X, prep) # (N, d_pca)
  
  # light AX augmentation
  A_feat <- row_norm(+A_raw); diag(A_feat) <- 1
  X_ax <- A_feat %*% Xp_all
  Xaug_all <- row_l2(cbind(Xp_all, 0.12 * X_ax))
  
  # ----- inputs -----
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
  
  # focal alpha from TRAIN distribution
  alpha_counts <- tabulate(y[train_idx] + 1L, nbins = Kc)
  inv_freq <- 1 / pmax(alpha_counts, 1)
  alpha_vec <- as.numeric(inv_freq / mean(inv_freq))
  
  # ----- build (YOUR PARAMS) -----
  model <- build_e2e_boost_flexible(
    n_nodes = N, n_features = ncol(Xp_all), num_classes = Kc,
    T = 3L,
    init_shrink = c(0.90, 0.85, 0.80),
    k_top = 120L, tau_cos = 0.2,
    init_alpha_blend = 0.05,
    block_feature_dim = 64L, prev_emb_dim = 16L, gate_dim_target = 96L,
    block_depth = 5L, block_dropout = 0.2,
    lr = 5e-3, l2 = 1e-4,
    use_focal = TRUE, focal_gamma = 2, focal_alpha = alpha_vec,
    use_cluster_gate = TRUE, gate_beta = 10, gate_normalize_inputs = TRUE
  )
  
  # ----- train (YOUR PARAMS) -----
  model <- train_e2e_boost(
    model,
    inputs_trainval = inputs_all,
    y_batch = y_batch,
    train_mask = sw_train,
    val_mask = sw_val,
    rounds = 5L, epochs_per_round = 200L,
    hard_thresh = 0.45, up_factor = 4.0,
    verbose = 1L, es_patience = 40L,
    rlrop = TRUE, rlrop_factor = 0.5, rlrop_patience = 7L, rlrop_min_lr = 1e-5
  )
  
  # ----- evaluate -----
  p_list <- predict(model, inputs_all, verbose = 0)
  P <- if (is.list(p_list)) p_list[[1]] else p_list
  if (length(dim(P)) == 2L) P <- array(P, dim = c(1, nrow(P), ncol(P)))
  P <- P[1, , ]
  
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

# -------------- SUMMARY --------------
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
    "- summary mean??sd CSV: ", file.path(out_root, 'summary_mean_sd.csv'), "\n", sep="")