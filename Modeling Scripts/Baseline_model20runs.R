suppressPackageStartupMessages({
  library(keras)
  library(data.table)
})

###########################################################
## 0) USER PATHS (EDIT THESE)
###########################################################

# Where your split folders live:
# base_runs_dir/seed_01/{train_idx.txt,val_idx.txt,test_idx.txt} ...
base_runs_dir <- "C:/Users/user/Desktop/Thesis Coding/runs/cora_e2eboost"

# Where to write outputs (csv results, etc.)
out_dir <- file.path(base_runs_dir, "baseline_outputs")

# Create output directory if missing
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# IMPORTANT: set TRUE if your txt indices are 0-based (0..N-1)
ADD_ONE <- FALSE

###########################################################
## 1) Helpers
###########################################################

parse_idx_col <- function(x) {
  if (is.numeric(x)) return(as.integer(x))
  if (is.na(x) || x == "") return(integer(0))
  parts <- strsplit(as.character(x), ";", fixed = TRUE)[[1]]
  parts <- trimws(parts)
  as.integer(parts[nzchar(parts)])
}

read_idx <- function(path, add_one = FALSE) {
  idx <- scan(path, what = integer(), quiet = TRUE)
  if (add_one) idx <- idx + 1L
  as.integer(idx)
}

# --- If you don't already have these from your main pipeline, keep them here ---
row_l2 <- function(M) {
  den <- sqrt(rowSums(M^2))
  den[den == 0] <- 1
  M / den
}

fit_feature_prep <- function(X_train, use_tfidf = TRUE, l2_rows = TRUE, pca_dim = 256L) {
  X_train <- as.matrix(X_train)
  storage.mode(X_train) <- "double"
  
  idf <- NULL
  if (isTRUE(use_tfidf)) {
    df <- colSums(X_train > 0)
    idf <- log((nrow(X_train) + 1) / (df + 1)) + 1
  }
  
  Xw <- if (!is.null(idf)) sweep(X_train, 2L, idf, `*`) else X_train
  if (isTRUE(l2_rows)) Xw <- row_l2(Xw)
  
  rnk <- max(1L, min(as.integer(pca_dim), ncol(Xw), nrow(Xw) - 1L))
  pca <- prcomp(Xw, center = TRUE, scale. = FALSE, rank. = rnk)
  
  list(
    idf = idf,
    l2 = l2_rows,
    pca = pca,
    dim = rnk,
    use_tfidf = !is.null(idf),
    ncol = ncol(X_train)
  )
}

apply_feature_prep <- function(X_new, prep) {
  X_new <- as.matrix(X_new)
  storage.mode(X_new) <- "double"
  stopifnot(ncol(X_new) == prep$ncol)
  
  Xf <- if (isTRUE(prep$use_tfidf)) sweep(X_new, 2L, prep$idf, `*`) else X_new
  if (isTRUE(prep$l2)) Xf <- row_l2(Xf)
  
  Xp <- predict(prep$pca, Xf)
  Xp[, seq_len(prep$dim), drop = FALSE]
}

###########################################################
## 2) Fit + evaluate on one split (includes test error)
###########################################################

fit_model_on_split <- function(model,
                               Xp_all, y_all,
                               train_idx, val_idx, test_idx,
                               epochs = 400L,
                               batch_size = 256L,
                               verbose = 0L) {
  stopifnot(is.matrix(Xp_all), length(y_all) == nrow(Xp_all))
  
  train_idx <- as.integer(train_idx)
  val_idx <- as.integer(val_idx)
  test_idx <- as.integer(test_idx)
  
  x_train <- Xp_all[train_idx, , drop = FALSE]
  y_train <- y_all[train_idx]
  x_val <- Xp_all[val_idx, , drop = FALSE]
  y_val <- y_all[val_idx]
  
  model %>% fit(
    x = x_train, y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_val, y_val),
    verbose = verbose
  )
  
  p_all <- model %>% predict(Xp_all, verbose = 0)
  pred_all <- max.col(p_all) - 1L
  
  acc_train <- mean(pred_all[train_idx] == y_all[train_idx])
  acc_val <- mean(pred_all[val_idx] == y_all[val_idx])
  acc_test <- mean(pred_all[test_idx] == y_all[test_idx])
  
  list(
    acc_train = acc_train,
    acc_val = acc_val,
    acc_test = acc_test,
    err_test = 1 - acc_test
  )
}

###########################################################
## 3) Baseline model builders (unchanged)
###########################################################

build_mlp_baseline <- function(n_features, num_classes,
                               hidden_dims = c(64L),
                               dropout = 0.5,
                               lr = 1e-3,
                               l2 = 5e-4) {
  inp <- layer_input(shape = n_features, name = "X")
  h <- inp
  for (hd in hidden_dims) {
    h <- h %>%
      layer_dense(units = hd, activation = "relu",
                  kernel_regularizer = regularizer_l2(l2)) %>%
      layer_dropout(rate = dropout)
  }
  out <- h %>%
    layer_dense(units = num_classes, activation = "softmax",
                kernel_regularizer = regularizer_l2(l2))
  
  model <- keras_model(inputs = inp, outputs = out)
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )
  model
}

build_baseline_mlp <- function(n_features, num_classes) build_mlp_baseline(n_features, num_classes, c(64L), 0.5, 1e-3, 5e-4)
build_baseline_mlp2 <- function(n_features, num_classes) build_mlp_baseline(n_features, num_classes, c(128L,64L), 0.5, 1e-3, 5e-4)
build_baseline_gcn <- function(n_features, num_classes) build_mlp_baseline(n_features, num_classes, c(64L), 0.6, 1e-3, 5e-3)
build_baseline_gcn2 <- function(n_features, num_classes) build_mlp_baseline(n_features, num_classes, c(64L,64L), 0.6, 1e-3, 5e-3)
build_baseline_gat <- function(n_features, num_classes) build_mlp_baseline(n_features, num_classes, c(128L), 0.6, 5e-4, 5e-4)

get_baseline_builder <- function(model_type) {
  force(model_type)
  function(n_features, num_classes) {
    switch(
      model_type,
      mlp = build_baseline_mlp(n_features, num_classes),
      mlp2 = build_baseline_mlp2(n_features, num_classes),
      gcn = build_baseline_gcn(n_features, num_classes),
      gcn2 = build_baseline_gcn2(n_features, num_classes),
      gat = build_baseline_gat(n_features, num_classes),
      stop(sprintf("Unknown baseline type '%s'", model_type))
    )
  }
}

###########################################################
## 4) Build runs CSV FROM FOLDERS (fixes ???where are dirs????)
###########################################################

build_runs_csv_from_folders <- function(base_runs_dir, out_csv, add_one = FALSE, n_seeds = 20L) {
  stopifnot(dir.exists(base_runs_dir))
  all_runs <- list()
  row_id <- 1L
  
  for (s in seq_len(n_seeds)) {
    seed_dir <- file.path(base_runs_dir, sprintf("seed_%02d", s))
    train_file <- file.path(seed_dir, "train_idx.txt")
    val_file <- file.path(seed_dir, "val_idx.txt")
    test_file <- file.path(seed_dir, "test_idx.txt")
    
    if (!file.exists(train_file) || !file.exists(val_file) || !file.exists(test_file)) {
      warning(sprintf("Missing split files for seed_%02d, skipping.", s))
      next
    }
    
    train_idx <- read_idx(train_file, add_one = add_one)
    val_idx <- read_idx(val_file, add_one = add_one)
    test_idx <- read_idx(test_file, add_one = add_one)
    
    if (!length(train_idx) || !length(val_idx) || !length(test_idx)) {
      warning(sprintf("Empty indices for seed_%02d, skipping.", s))
      next
    }
    
    all_runs[[row_id]] <- data.frame(
      run_id = s,
      seed = s,
      train_idx = paste(train_idx, collapse = ";"),
      val_idx = paste(val_idx, collapse = ";"),
      test_idx = paste(test_idx, collapse = ";"),
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }
  
  runs_df <- do.call(rbind, all_runs)
  write.csv(runs_df, out_csv, row.names = FALSE)
  cat("??? Wrote runs CSV to:", out_csv, "\n")
  out_csv
}

runs_csv_path <- file.path(out_dir, "runs_baselines.csv")
runs_csv_path <- build_runs_csv_from_folders(base_runs_dir, runs_csv_path, add_one = ADD_ONE, n_seeds = 20L)

###########################################################
## 5) Run baselines (PER-RUN preprocessing INSIDE)
###########################################################

run_baselines_from_runs_per_runprep <- function(
    X_raw_all, y_all,
    runs_csv,
    model_types = c("mlp", "mlp2", "gcn", "gcn2", "gat"),
    epochs = 400L,
    batch_size = 256L,
    verbose = 0L,
    use_tfidf = TRUE,
    l2_rows = TRUE,
    pca_dim = 256L
) {
  stopifnot(is.matrix(X_raw_all), length(y_all) == nrow(X_raw_all))
  
  runs <- read.csv(runs_csv, stringsAsFactors = FALSE, check.names = FALSE)
  required_cols <- c("run_id","seed","train_idx","val_idx","test_idx")
  if (!all(required_cols %in% colnames(runs))) {
    stop("runs_csv must contain columns: run_id, seed, train_idx, val_idx, test_idx")
  }
  
  Kc <- length(unique(y_all))
  results <- list()
  row_id <- 1L
  
  for (mt in model_types) {
    cat(sprintf("\n=== Baseline '%s' over %d runs (per-run TFIDF/L2/PCA) ===\n", mt, nrow(runs)))
    builder_fn <- get_baseline_builder(mt)
    
    for (i in seq_len(nrow(runs))) {
      r <- runs[i, ]
      run_id <- r$run_id
      seed <- r$seed
      
      train_idx <- parse_idx_col(r$train_idx)
      val_idx <- parse_idx_col(r$val_idx)
      test_idx <- parse_idx_col(r$test_idx)
      
      if (!length(train_idx) || !length(val_idx) || !length(test_idx)) next
      
      cat(sprintf(" -> %s | run_id=%s seed=%s | N_train=%d N_val=%d N_test=%d\n",
                  mt, run_id, seed, length(train_idx), length(val_idx), length(test_idx)))
      
      set.seed(seed)
      if (exists("k_clear_session", mode = "function")) k_clear_session()
      
      # PER-RUN prep
      pca_dim_run <- min(as.integer(pca_dim), length(train_idx) - 1L)
      if (!is.finite(pca_dim_run) || pca_dim_run < 1L) pca_dim_run <- 1L
      
      prep <- fit_feature_prep(
        X_train = X_raw_all[train_idx, , drop = FALSE],
        use_tfidf = use_tfidf,
        l2_rows = l2_rows,
        pca_dim = pca_dim_run
      )
      Xp_all_run <- apply_feature_prep(X_raw_all, prep)
      
      model <- builder_fn(ncol(Xp_all_run), Kc)
      
      fit_out <- fit_model_on_split(
        model = model,
        Xp_all = Xp_all_run,
        y_all = y_all,
        train_idx = train_idx,
        val_idx = val_idx,
        test_idx = test_idx,
        epochs = epochs,
        batch_size = batch_size,
        verbose = verbose
      )
      
      results[[row_id]] <- data.frame(
        model_type = mt,
        run_id = run_id,
        seed = seed,
        N_train = length(train_idx),
        N_val = length(val_idx),
        N_test = length(test_idx),
        acc_train = fit_out$acc_train,
        acc_val = fit_out$acc_val,
        acc_test = fit_out$acc_test,
        err_test = fit_out$err_test,
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }
  
  do.call(rbind, results)
}

###########################################################
## 6) RUN (you must provide RAW X and y in your environment)
###########################################################

# REQUIRED objects in your workspace before running:
# X <- raw feature matrix (N x Fraw)
# y <- integer labels 0..K-1 length N
stopifnot(exists("X"), exists("y"))
X <- as.matrix(X)
y <- as.integer(y)

res_baselines <- run_baselines_from_runs_per_runprep(
  X_raw_all = X,
  y_all = y,
  runs_csv = runs_csv_path,
  model_types = c("mlp","mlp2","gcn","gcn2","gat"),
  epochs = 400L,
  batch_size = 256L,
  verbose = 0L,
  pca_dim = 256L
)

out_csv <- file.path(out_dir, "baseline_results.csv")
write.csv(res_baselines, out_csv, row.names = FALSE)
cat("??? Wrote baseline results to:", out_csv, "\n")

res_dt <- as.data.table(res_baselines)
print(res_dt[, .(
  mean_acc_test = mean(acc_test),
  sd_acc_test = sd(acc_test),
  mean_err_test = mean(err_test),
  sd_err_test = sd(err_test)
), by = model_type])
